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
)


def test_center_slice_is_found():
    _, lab, cz = make_case()
    assert find_center_slice(lab) == cz


def test_center_slice_multi_slice_label_warns_and_averages(capsys):
    _, lab, cz = make_case()
    arr = sitk.GetArrayFromImage(lab)
    arr[cz + 1] = arr[cz]                       # label now spans two slices
    lab2 = sitk.GetImageFromArray(arr)
    lab2.CopyInformation(lab)
    idx = find_center_slice(lab2)
    assert idx in (cz, cz + 1)
    assert "spans 2 slices" in capsys.readouterr().out


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
