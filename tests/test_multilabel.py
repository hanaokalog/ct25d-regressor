"""Multi-label files: select one structure by value, and only that one.

The load-bearing test is `test_touching_label_does_not_bleed_in`. Resampling a
multi-label image and thresholding afterwards -- the obvious implementation --
pulls in every neighbouring structure whose value clears the threshold, and it
does so silently, because the result still looks like a plausible mask.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pd = pytest.importorskip("pandas")

from ct25d.crop import crop_pair  # noqa: E402
from ct25d.geometry import (  # noqa: E402
    available_labels,
    binarize_label,
    build_sample_sitk,
    find_center_slice,
    mask_area_mm2,
    mask_centroid_index,
)

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
INPLANE = 0.78125


def multilabel_case(directory=None, spacing=INPLANE, z_spacing=2.5,
                    size=(160, 160, 20), dtype=np.uint8):
    """
    Three structures on the centre slice:
      label 2  a disc at (80, 60)          <- the usual target
      label 4  a disc at (80, 76), TOUCHING label 2
      label 7  a disc far away at (30, 130)
    """
    W, H, Z = size
    cz = Z // 2
    yy, xx = np.mgrid[0:H, 0:W]
    vol = np.stack([np.full((H, W), -100.0 + 20.0 * z, np.float32)
                    for z in range(Z)])
    lab = np.zeros((Z, H, W), dtype)
    d2 = ((yy - 80) ** 2 + (xx - 60) ** 2) <= 8.0 ** 2
    d4 = ((yy - 80) ** 2 + (xx - 76) ** 2) <= 8.0 ** 2
    d7 = ((yy - 30) ** 2 + (xx - 130) ** 2) <= 5.0 ** 2
    lab[cz][d2] = 2
    lab[cz][d4] = 4
    lab[cz][d7] = 7
    for z in range(Z):                        # some intensity to look at
        vol[z] += 600.0 * (d2 | d4 | d7)

    img, mask = sitk.GetImageFromArray(vol), sitk.GetImageFromArray(lab)
    for it in (img, mask):
        it.SetSpacing((spacing, spacing, z_spacing))
        it.SetOrigin((-40.0, -50.0, 5.0))
    # count from the written array: the discs of label 2 and 4 touch, so the
    # later assignment takes a pixel off label 2
    counts = {v: int((lab == v).sum()) for v in (2, 4, 7)}
    if directory is None:
        return img, mask, counts
    ip = Path(directory) / "ml_img.nii.gz"
    mp = Path(directory) / "ml_seg.nii.gz"
    sitk.WriteImage(img, str(ip), True)
    sitk.WriteImage(mask, str(mp), True)
    return ip, mp, counts


def test_available_labels():
    _, mask, _ = multilabel_case()
    assert available_labels(mask) == [2, 4, 7]


@pytest.mark.parametrize("value", [2, 4, 7])
def test_binarize_selects_one_label(value):
    _, mask, counts = multilabel_case()
    b = binarize_label(mask, value)
    arr = sitk.GetArrayViewFromImage(b)
    assert set(np.unique(arr).tolist()) <= {0, 1}
    assert int(arr.sum()) == counts[value]


@pytest.mark.parametrize("value", [2, 4, 7])
def test_area_and_centroid_follow_the_label_value(value):
    _, mask, counts = multilabel_case()
    assert mask_area_mm2(mask, value) == pytest.approx(counts[value] * INPLANE ** 2)
    assert find_center_slice(mask, value) == 10
    cx, cy, _ = mask_centroid_index(mask, value)
    expected = {2: (60, 80), 4: (76, 80), 7: (130, 30)}[value]
    assert (cx, cy) == pytest.approx(expected, abs=0.6)


def test_touching_label_does_not_bleed_in():
    """
    Label 4 touches label 2. Resampling the raw label image and thresholding at
    0.5 * label_value afterwards would accept every voxel of value >= 1, which
    is both discs plus the interpolated values between them.
    """
    img, mask, counts = multilabel_case()
    s = build_sample_sitk(img, mask, label_value=2, crop_size=96)
    area = float(s[3].sum()) * INPLANE ** 2
    assert area == pytest.approx(counts[2] * INPLANE ** 2, rel=0.05)
    assert area < 0.6 * (counts[2] + counts[4]) * INPLANE ** 2

    # nothing survives where label 4 sits: 16 px to the right of label 2's centre
    yy, xx = np.nonzero(s[3])
    cx = xx.mean()
    assert xx.max() - cx < 12          # one disc's radius, not two discs apart


def test_each_label_gives_its_own_sample():
    img, mask, counts = multilabel_case()
    a = build_sample_sitk(img, mask, label_value=2, crop_size=64)
    b = build_sample_sitk(img, mask, label_value=7, crop_size=64)
    assert not np.array_equal(a[3], b[3])
    assert float(a[3].sum()) > float(b[3].sum())        # label 7 is smaller
    assert float(b[3].sum()) == pytest.approx(counts[7], rel=0.1)


def test_float_typed_label_file_still_matches():
    """Label images are sometimes stored as float; 3.0000001 == 3 is False."""
    img, mask, counts = multilabel_case(dtype=np.float32)
    assert mask_area_mm2(mask, 4) == pytest.approx(counts[4] * INPLANE ** 2)
    s = build_sample_sitk(img, mask, label_value=4, crop_size=64)
    assert float(s[3].sum()) == pytest.approx(counts[4], rel=0.1)


def test_missing_label_names_the_ones_present():
    img, mask, _ = multilabel_case()
    with pytest.raises(ValueError, match=r"labels present: \[2, 4, 7\]"):
        build_sample_sitk(img, mask, label_value=3)


# --------------------------------------------------------------------------- #
def test_crop_uses_only_the_requested_label(tmp_path):
    ip, mp, counts = multilabel_case(tmp_path)
    stats = crop_pair(ip, mp, tmp_path / "c_img.nii.gz", tmp_path / "c_seg.nii.gz",
                      label_value=7, margin_mm=15.0, margin_mm_z=10.0)
    assert stats["label_voxels"] == counts[7]
    assert stats["labels_present"] == [2, 4, 7]
    crop = sitk.ReadImage(str(tmp_path / "c_seg.nii.gz"))
    arr = sitk.GetArrayViewFromImage(crop)
    # the box is drawn around label 7, so the distant discs fall outside it
    assert int((arr == 7).sum()) == counts[7]
    assert int((arr == 2).sum()) == 0


def test_crop_keeps_other_labels_when_they_are_inside_the_box(tmp_path):
    ip, mp, counts = multilabel_case(tmp_path)
    crop_pair(ip, mp, tmp_path / "a_img.nii.gz", tmp_path / "a_seg.nii.gz",
              label_value=2, margin_mm=30.0)
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "a_seg.nii.gz")))
    assert int((arr == 2).sum()) == counts[2]
    assert int((arr == 4).sum()) == counts[4]      # the neighbour is preserved

    crop_pair(ip, mp, tmp_path / "b_img.nii.gz", tmp_path / "b_seg.nii.gz",
              label_value=2, margin_mm=30.0, binarize=True)
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(tmp_path / "b_seg.nii.gz")))
    assert set(np.unique(arr).tolist()) <= {0, 1}
    assert int(arr.sum()) == counts[2]


def test_cropped_multilabel_gives_the_same_sample(tmp_path):
    ip, mp, _ = multilabel_case(tmp_path)
    crop_pair(ip, mp, tmp_path / "c_img.nii.gz", tmp_path / "c_seg.nii.gz",
              label_value=2, margin_mm=40.0, margin_mm_z=10.0)
    a = build_sample_sitk(sitk.ReadImage(str(ip)), sitk.ReadImage(str(mp)),
                          label_value=2, crop_size=48)
    b = build_sample_sitk(sitk.ReadImage(str(tmp_path / "c_img.nii.gz")),
                          sitk.ReadImage(str(tmp_path / "c_seg.nii.gz")),
                          label_value=2, crop_size=48)
    assert np.array_equal(a[3], b[3])
    rel = np.abs(a[:-1] - b[:-1]).max() / max(float(np.abs(a[:-1]).max()), 1.0)
    assert rel < 1e-5


def test_cli_label_value_end_to_end(tmp_path):
    ip, mp, counts = multilabel_case(tmp_path)
    csv = tmp_path / "in.csv"
    pd.DataFrame([{"ct": str(ip), "seg": str(mp), "case_id": "m0",
                   "area_mm2": counts[7] * INPLANE ** 2}]).to_csv(csv, index=False)
    out_csv = tmp_path / "out.csv"
    res = subprocess.run(
        [sys.executable, str(EXAMPLES / "crop.py"), str(csv), str(out_csv),
         "--out-dir", str(tmp_path / "crops"), "--image-col", "ct",
         "--mask-col", "seg", "--id-col", "case_id",
         "--label-value", "7", "--binarize-mask", "--margin-mm", "15"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "target label value: 7" in res.stdout
    assert "written as 0/1" in res.stdout

    row = pd.read_csv(out_csv).iloc[0]
    arr = sitk.GetArrayFromImage(sitk.ReadImage(row["mask_crop"]))
    assert set(np.unique(arr).tolist()) <= {0, 1}
    assert int(arr.sum()) == counts[7]
