"""Geometry: does the right plane, at the right physical position, come out."""

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from conftest import make_case

from ct25d.constants import PIXEL_AREA_MM2
from ct25d.geometry import (
    build_sample_sitk,
    build_samples_sitk,
    find_center_slice,
    mask_area_mm2,
    slab_offsets_mm,
)


def test_center_slice_is_found():
    _, lab, cz = make_case()
    assert find_center_slice(lab) == cz


def test_center_slice_multi_slice_label_warns_and_takes_largest(capsys):
    _, lab, cz = make_case()
    arr = sitk.GetArrayFromImage(lab)
    arr[cz + 1] = arr[cz]                       # label now spans two slices
    lab2 = sitk.GetImageFromArray(arr)
    lab2.CopyInformation(lab)
    idx = find_center_slice(lab2)
    assert idx in (cz, cz + 1)
    assert "spans 2 slices" in capsys.readouterr().out


def test_center_slice_with_a_gap_never_lands_on_an_empty_slice():
    # labels on z = cz - 1 and cz + 1: a rounded centroid gives cz, which is empty
    _, lab, cz = make_case()
    arr = sitk.GetArrayFromImage(lab)
    small = np.zeros_like(arr[cz])
    small[arr[cz] > 0] = 1
    small[: small.shape[0] // 2] = 0            # half the area
    arr[cz - 1], arr[cz + 1], arr[cz] = small, arr[cz], 0
    lab2 = sitk.GetImageFromArray(arr)
    lab2.CopyInformation(lab)
    assert find_center_slice(lab2) == cz + 1


def test_planes_come_from_the_expected_source_slices():
    # native 2.5 mm, 5 mm gap -> cz-2, cz, cz+2 -> HU 1000 / 1200 / 1400
    img, lab, cz = make_case(z_spacing=2.5)
    s = build_sample_sitk(img, lab, center_on_mask=False)
    assert [round(float(np.median(p))) for p in s[:3]] == [1000, 1200, 1400]


def test_output_pixel_size_is_the_target():
    img, lab, _ = make_case(size=(200, 200, 24), inplane=0.9766)
    s = build_sample_sitk(img, lab, center_on_mask=False)
    assert s.shape == (4, 250, 250)             # 200 * 0.9766 / 0.78125


def test_flipped_direction_cosines_give_the_same_planes():
    a = build_sample_sitk(*make_case()[:2], center_on_mask=False)
    b = build_sample_sitk(*make_case(direction_flip=True)[:2], center_on_mask=False)
    assert [round(float(np.median(p))) for p in a[:3]] == \
           [round(float(np.median(p))) for p in b[:3]]
    assert abs(float(a[3].sum()) - float(b[3].sum())) / float(a[3].sum()) < 0.01


def test_resampled_mask_area_matches_the_source():
    img, lab, _ = make_case()
    s = build_sample_sitk(img, lab, center_on_mask=False)
    ratio = float(s[3].sum()) * PIXEL_AREA_MM2 / mask_area_mm2(lab)
    assert 0.97 < ratio < 1.03


def test_crop_is_centred_on_the_structure():
    img, lab, _ = make_case(center_yx=(60.0, 130.0), radius_px=9.0)
    s = build_sample_sitk(img, lab, crop_size=96, center_on_mask=True)
    assert s.shape == (4, 96, 96)
    yy, xx = np.nonzero(s[3])
    assert abs(yy.mean() - 47.5) < 1.5 and abs(xx.mean() - 47.5) < 1.5


def test_border_slice_duplicates_its_neighbour_instead_of_air():
    img, lab, _ = make_case(z_spacing=5.0, size=(120, 120, 12), center_index=0)
    s = build_sample_sitk(img, lab, crop_size=64)
    med = [round(float(np.median(p))) for p in s[:3]]
    assert med == [0, 0, 100]                   # clamped, not filled with air


def test_mask_on_a_different_grid_is_resampled_onto_the_image():
    img, lab, cz = make_case(inplane=0.9766)
    arr = sitk.GetArrayFromImage(lab)
    small = sitk.GetImageFromArray(arr[:, ::2, ::2])   # coarser mask grid
    small.SetSpacing((0.9766 * 2, 0.9766 * 2, lab.GetSpacing()[2]))
    small.SetOrigin(lab.GetOrigin())
    s = build_sample_sitk(img, small, crop_size=96)
    assert s[3].sum() > 0
    yy, xx = np.nonzero(s[3])
    assert abs(yy.mean() - 47.5) < 2.0 and abs(xx.mean() - 47.5) < 2.0


def test_batch_requires_a_common_output_size():
    a = make_case(size=(160, 160, 10), inplane=0.7, z_spacing=5.0)
    b = make_case(size=(200, 220, 12), inplane=0.9, z_spacing=5.0)
    out = build_samples_sitk([a[0], b[0]], [a[1], b[1]], crop_size=96)
    assert out.shape == (2, 4, 96, 96)
    with pytest.raises(ValueError, match="inconsistent output shapes"):
        build_samples_sitk([a[0], b[0]], [a[1], b[1]])


def test_empty_mask_is_rejected():
    img, lab, _ = make_case()
    shape = sitk.GetArrayFromImage(lab).shape
    empty = sitk.GetImageFromArray(np.zeros(shape, np.uint8))
    empty.CopyInformation(lab)
    with pytest.raises(ValueError, match="no voxel with label"):
        build_sample_sitk(img, empty)


# ---- slab averaging ------------------------------------------------------- #

def test_slab_sample_count_follows_the_native_spacing():
    assert slab_offsets_mm(5.0, 1.0).tolist() == [-2.0, -1.0, 0.0, 1.0, 2.0]
    assert slab_offsets_mm(5.0, 2.5).tolist() == [-1.25, 1.25]
    assert slab_offsets_mm(5.0, 5.0).tolist() == [0.0]
    assert slab_offsets_mm(5.0, 7.0).tolist() == [0.0]
    assert slab_offsets_mm(0.0, 1.0).tolist() == [0.0]
    assert slab_offsets_mm(None, 1.0).tolist() == [0.0]


def test_slab_leaves_5mm_volumes_unchanged():
    img, lab, _ = make_case(z_spacing=5.0, size=(120, 120, 12))
    a = build_sample_sitk(img, lab, crop_size=64, slab_mm=5.0)
    b = build_sample_sitk(img, lab, crop_size=64, slab_mm=0.0)
    assert np.array_equal(a, b)


def _acquire(thickness_mm, n_slices, period_mm=8.0, amp=500.0, size=64):
    """
    The same anatomy scanned at a given slice thickness: every slice holds the
    mean of f(z) = amp * sin(2 pi z / period) over its own thickness, which is
    what a scanner's slice profile does. Slices are contiguous from z = 0.
    """
    vol = np.empty((n_slices, size, size), np.float32)
    for k in range(n_slices):
        u = (np.arange(400) + 0.5) / 400 - 0.5          # midpoint rule
        zz = k * thickness_mm + u * thickness_mm
        vol[k] = amp * np.sin(2 * np.pi * zz / period_mm).mean()
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((0.8, 0.8, thickness_mm))
    yy, xx = np.mgrid[0:size, 0:size]
    msk = np.zeros(vol.shape, np.uint8)
    msk[int(round(30.0 / thickness_mm))] = (yy - 32) ** 2 + (xx - 32) ** 2 <= 100
    lab = sitk.GetImageFromArray(msk)
    lab.CopyInformation(img)
    return img, lab


def test_thin_slices_averaged_over_5mm_match_a_5mm_scan():
    thick = build_sample_sitk(*_acquire(5.0, 12), crop_size=32)
    thin = build_sample_sitk(*_acquire(1.0, 60), crop_size=32)
    raw = build_sample_sitk(*_acquire(1.0, 60), crop_size=32, slab_mm=0.0)

    err_slab = np.abs(thin[:3] - thick[:3]).max()
    err_raw = np.abs(raw[:3] - thick[:3]).max()
    assert err_slab < 1.0                        # HU, against an amplitude of 500
    assert err_raw > 100.0                       # what the 4 mm in between carry
    assert np.array_equal(thin[3], thick[3])     # the mask is not averaged


def test_slab_at_the_volume_border_clamps_instead_of_reading_air():
    img, lab, _ = make_case(z_spacing=1.0, size=(120, 120, 30), center_index=0)
    s = build_sample_sitk(img, lab, crop_size=64)
    # centre plane: samples at -2..2 mm, the negative ones clamp to slice 0
    assert round(float(np.median(s[1]))) == round((0 + 0 + 0 + 100 + 200) / 5)
    assert float(s[:3].min()) >= 0.0


def test_planes_follow_the_index_order_not_the_header(case):
    """with_direction changes the physical frame; the 2.5D planes do not care."""
    from ct25d.geometry import build_sample_sitk, with_direction

    img, lab, cz = case                       # slice k holds 100 * k HU
    flipped = with_direction(img, lab, (1, 0, 0, 0, 1, 0, 0, 0, -1))
    assert flipped[0].GetDirection()[8] == -1.0 and img.GetDirection()[8] == 1.0
    a = build_sample_sitk(img, lab, n_slices=3, gap_mm=2.5, slab_mm=0, crop_size=32)
    b = build_sample_sitk(*flipped, n_slices=3, gap_mm=2.5, slab_mm=0, crop_size=32)
    assert np.allclose(a, b)
    assert a[0, 16, 16] == 100.0 * (cz - 1) and a[2, 16, 16] == 100.0 * (cz + 1)
    assert with_direction(img, lab, None) == (img, lab)


def test_direction_override_needs_one_grid(case):
    from ct25d.geometry import with_direction

    img, lab, _ = case
    other = sitk.Image(lab)
    other.SetOrigin((0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        with_direction(img, other, (1, 0, 0, 0, 1, 0, 0, 0, -1))
