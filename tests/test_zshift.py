"""z-shift augmentation: thin-slice volumes get image planes built at small
through-plane shifts; the mask stays on the labelled slice."""

import numpy as np
import pandas as pd
import pytest
import torch

sitk = pytest.importorskip("SimpleITK")

from conftest import make_case

from ct25d.data import SliceStackDataset
from ct25d.geometry import build_sample_sitk
from ct25d.tabular import load_stack_cache, prepare_stacks, save_stack_cache


def test_shift_moves_the_image_planes_but_not_the_mask():
    # slice k is filled with 100 * k; 1 mm spacing, so 1 mm is 100 HU
    img, lab, cz = make_case(size=(64, 64, 40), z_spacing=1.0, center_index=20)
    base = build_sample_sitk(img, lab, crop_size=32, slab_mm=0)
    for d in (-2.0, -1.0, 1.0, 2.0):
        s = build_sample_sitk(img, lab, crop_size=32, slab_mm=0, z_shift_mm=d)
        assert np.allclose(s[:3] - base[:3], 100.0 * d, atol=1e-3)
        assert np.array_equal(s[3], base[3])
    # with the 5 mm slab the mean moves by the same amount
    a = build_sample_sitk(img, lab, crop_size=32)
    b = build_sample_sitk(img, lab, crop_size=32, z_shift_mm=2.0)
    assert np.allclose(b[:3] - a[:3], 200.0, atol=1e-3)


def write(tmp_path, name, z_spacing):
    img, lab, _ = make_case(size=(64, 64, 40), z_spacing=z_spacing, center_index=20)
    ip, mp = tmp_path / f"{name}_img.nii.gz", tmp_path / f"{name}_seg.nii.gz"
    sitk.WriteImage(img, str(ip))
    sitk.WriteImage(lab, str(mp))
    return dict(image=str(ip), mask=str(mp))


def test_only_thin_slices_get_variants_and_the_cache_keeps_them(tmp_path):
    df = pd.DataFrame([write(tmp_path, "thin", 1.0), write(tmp_path, "thick", 5.0),
                       write(tmp_path, "mid", 2.5)])
    shifts = (-2.0, -1.0, 1.0, 2.0)
    stacks, kept, _, var = prepare_stacks(df, crop_size=32, verbose=False,
                                          z_shifts_mm=shifts)
    plain = prepare_stacks(df, crop_size=32, verbose=False)
    assert len(plain) == 3 and np.array_equal(stacks, plain[0])
    assert var[1] is None
    for i in (0, 2):
        assert var[i].shape == (4, 3, 32, 32)
    # 1 mm spacing: slice k holds 100 k, so a 2 mm shift is +200 HU
    assert np.allclose(var[0][3] - stacks[0][:3], 200.0, atol=1e-3)

    prep = dict(crop_size=32, z_shifts_mm=list(shifts))
    path = tmp_path / "c.npz"
    save_stack_cache(path, stacks, kept, prep, variants=var)
    s, k, v = load_stack_cache(path, prep, with_variants=True)
    assert np.array_equal(s, stacks) and v[1] is None
    assert np.array_equal(v[0], var[0]) and np.array_equal(v[2], var[2])
    s2, k2 = load_stack_cache(path, prep)                 # old call still works
    assert np.array_equal(s2, stacks)


def test_dataset_draws_every_shift_and_keeps_the_mask(stack_hu):
    var = np.stack([stack_hu[:3] + d for d in (-200.0, -100.0, 100.0, 200.0)])
    ds = SliceStackDataset([stack_hu, stack_hu], [0, 1], standardizer=None,
                           task="classification", mask_channel="binary",
                           window=(-1000, 1000), z_variants=[var, None])
    seen = set()
    torch.manual_seed(0)
    for _ in range(200):
        x, _ = ds[0]
        assert torch.equal(x[3], torch.as_tensor(stack_hu[3]))   # mask channel
        seen.add(round(float(x[0, 48, 48]), 3))                   # windowed value
        y, _ = ds[1]
        assert torch.allclose(y[:3], ds[1][0][:3])               # no variants: fixed
    assert len(seen) == 5
    with pytest.raises(ValueError):
        SliceStackDataset([stack_hu], [0], standardizer=None, task="classification",
                          z_variants=[None, None])
