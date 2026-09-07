"""Cropping: does the crop preserve everything the pipeline needs.

The load-bearing test is `test_pipeline_output_is_unchanged_by_cropping` --
`build_sample_sitk` must produce the same array from the cropped pair as from
the original. Everything else is a way of localizing a failure of that one.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
pd = pytest.importorskip("pandas")

from ct25d.crop import crop_pair, label_bbox_physical, required_margin_mm  # noqa: E402
from ct25d.geometry import build_sample_sitk, mask_area_mm2  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def make_big_case(directory, i=0, inplane=0.72, z_spacing=1.25,
                  size=(256, 256, 60), label_at=(180, 70, 33), radius_px=7.0,
                  mask_inplane=None, empty=False):
    """A large volume with a small off-centre label, as in a real study."""
    W, H, Z = size
    yy, xx = np.mgrid[0:H, 0:W]
    rng = np.random.default_rng(i)
    vol = rng.normal(-200, 60, (Z, H, W)).astype(np.float32)
    cx, cy, cz = label_at
    disc = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px ** 2)
    for z in range(Z):
        shrink = max(0.0, 1.0 - abs(z - cz) / 5.0)
        vol[z] += 900.0 * (((yy - cy) ** 2 + (xx - cx) ** 2)
                           <= (radius_px * shrink) ** 2)

    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((inplane, inplane, z_spacing))
    img.SetOrigin((-150.0, -180.0, 12.0))

    if mask_inplane is None:
        mask = np.zeros((Z, H, W), np.uint8)
        if not empty:
            mask[cz] = disc
        lab = sitk.GetImageFromArray(mask)
        lab.CopyInformation(img)
    else:                                   # label on its own coarser grid
        f = int(round(mask_inplane / inplane))
        mask = np.zeros((Z, H // f, W // f), np.uint8)
        if not empty:
            mask[cz] = disc[::f, ::f]
        lab = sitk.GetImageFromArray(mask)
        lab.SetSpacing((inplane * f, inplane * f, z_spacing))
        lab.SetOrigin(img.GetOrigin())

    ip = Path(directory) / f"big{i}_img.nii.gz"
    mp = Path(directory) / f"big{i}_seg.nii.gz"
    sitk.WriteImage(img, str(ip), True)
    sitk.WriteImage(lab, str(mp), True)
    return ip, mp


def test_required_margin_covers_the_augmented_patch():
    m = required_margin_mm(96, 0.78125, scale_min=0.9, rotate_deg=5.0,
                           translate=0.08)
    half = 0.5 * 96 * 0.78125                       # 37.5 mm, no augmentation
    assert m > half                                  # zoom-out and rotation add
    assert m == required_margin_mm(96)               # defaults match
    assert required_margin_mm(48) < required_margin_mm(96)
    assert required_margin_mm(96, scale_min=0.5) > required_margin_mm(96)


def test_bbox_margin_is_physical_not_voxels(tmp_path):
    _, m1 = make_big_case(tmp_path, i=1, inplane=0.72)
    _, m2 = make_big_case(tmp_path, i=2, inplane=0.98, size=(200, 200, 60),
                          label_at=(140, 60, 33), radius_px=7.0 * 0.72 / 0.98)
    c1 = np.array(label_bbox_physical(sitk.ReadImage(str(m1)), margin_mm=30.0))
    c2 = np.array(label_bbox_physical(sitk.ReadImage(str(m2)), margin_mm=30.0))
    e1 = c1.max(axis=0) - c1.min(axis=0)
    e2 = c2.max(axis=0) - c2.min(axis=0)
    assert np.allclose(e1[:2], e2[:2], atol=2.0)     # same physical extent


def test_crop_reduces_the_volume_and_keeps_every_label_voxel(tmp_path):
    ip, mp = make_big_case(tmp_path)
    stats = crop_pair(ip, mp, tmp_path / "c_img.nii.gz", tmp_path / "c_seg.nii.gz",
                      margin_mm=40.0, margin_mm_z=10.0)
    assert stats["reduction"] > 5.0
    assert stats["image_voxels_after"] < stats["image_voxels_before"]
    cropped = sitk.ReadImage(str(tmp_path / "c_seg.nii.gz"))
    n_kept = np.count_nonzero(sitk.GetArrayViewFromImage(cropped))
    assert n_kept == stats["label_voxels"]
    assert mask_area_mm2(cropped) == pytest.approx(
        mask_area_mm2(sitk.ReadImage(str(mp))))


def test_crop_preserves_physical_position(tmp_path):
    ip, mp = make_big_case(tmp_path)
    crop_pair(ip, mp, tmp_path / "c_img.nii.gz", tmp_path / "c_seg.nii.gz")
    full, crop = sitk.ReadImage(str(ip)), sitk.ReadImage(str(tmp_path / "c_img.nii.gz"))
    # a physical point inside the crop must map to the same voxel value
    p = crop.TransformIndexToPhysicalPoint((5, 6, 2))
    i_full = full.TransformPhysicalPointToIndex(p)
    assert (sitk.GetArrayViewFromImage(crop)[2, 6, 5]
            == pytest.approx(sitk.GetArrayViewFromImage(full)[i_full[2], i_full[1],
                                                              i_full[0]], abs=1e-4))


@pytest.mark.parametrize("mask_inplane", [None, 1.44])
def test_pipeline_output_is_unchanged_by_cropping(tmp_path, mask_inplane):
    """The whole point of the tool: the model must not be able to tell."""
    ip, mp = make_big_case(tmp_path, mask_inplane=mask_inplane)
    cip, cmp_ = tmp_path / "c_img.nii.gz", tmp_path / "c_seg.nii.gz"
    crop_pair(ip, mp, cip, cmp_,
              margin_mm=required_margin_mm(64), margin_mm_z=10.0)

    a = build_sample_sitk(sitk.ReadImage(str(ip)), sitk.ReadImage(str(mp)),
                          crop_size=64)
    b = build_sample_sitk(sitk.ReadImage(str(cip)), sitk.ReadImage(str(cmp_)),
                          crop_size=64)
    assert a.shape == b.shape
    # the mask channel must be identical; the image channels differ only by
    # float32 rounding, since the resampler now works from a different origin
    assert np.array_equal(a[-1], b[-1])
    rel = np.abs(a[:-1] - b[:-1]).max() / max(float(np.abs(a[:-1]).max()), 1.0)
    assert rel < 1e-5, rel


def test_empty_label_is_rejected(tmp_path):
    ip, mp = make_big_case(tmp_path, i=9, empty=True)
    with pytest.raises(ValueError, match="no voxel with label"):
        crop_pair(ip, mp, tmp_path / "x.nii.gz", tmp_path / "y.nii.gz")


# --------------------------------------------------------------------------- #
def _cohort(tmp_path, n=4, with_empty=True):
    rows = []
    for i in range(n):
        empty = with_empty and i == n - 1
        ip, mp = make_big_case(tmp_path, i=i, empty=empty,
                               label_at=(150 + 10 * i, 70, 30 + i))
        rows.append({"ct": str(ip), "seg": str(mp), "case_id": f"c{i}",
                     "area_mm2": 100.0 + i})
    csv = tmp_path / "in.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def run_cli(argv):
    out = subprocess.run([sys.executable, str(EXAMPLES / "crop.py"), *argv],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_cli_writes_paths_and_drops_failures(tmp_path):
    csv = _cohort(tmp_path)
    out_csv = tmp_path / "out.csv"
    log = run_cli([str(csv), str(out_csv), "--out-dir", str(tmp_path / "crops"),
                   "--image-col", "ct", "--mask-col", "seg",
                   "--id-col", "case_id", "--for-crop-size", "64"])
    assert "volume reduction" in log
    df = pd.read_csv(out_csv)
    assert len(df) == 3                               # the empty label was dropped
    assert "image_crop" in df.columns and "mask_crop" in df.columns
    assert "area_mm2" in df.columns                   # original columns survive
    for _, row in df.iterrows():
        assert Path(row["image_crop"]).exists()
        assert Path(row["mask_crop"]).exists()
        assert "c" in Path(row["image_crop"]).name    # named from --id-col


def test_cli_keep_failed_and_skip_existing(tmp_path):
    csv = _cohort(tmp_path)
    out_csv = tmp_path / "out.csv"
    args = [str(csv), str(out_csv), "--out-dir", str(tmp_path / "crops"),
            "--image-col", "ct", "--mask-col", "seg", "--keep-failed"]
    run_cli(args)
    df = pd.read_csv(out_csv)
    assert len(df) == 4 and "crop_ok" in df.columns
    assert bool(df["crop_ok"].iloc[-1]) is False
    assert pd.isna(df["image_crop"].iloc[-1]) or df["image_crop"].iloc[-1] == ""

    log = run_cli(args + ["--skip-existing"])
    assert "reused 3" in log


def test_cli_parallel_matches_serial(tmp_path):
    csv = _cohort(tmp_path, with_empty=False)
    a, b = tmp_path / "a.csv", tmp_path / "b.csv"
    common = ["--image-col", "ct", "--mask-col", "seg", "--id-col", "case_id"]
    run_cli([str(csv), str(a), "--out-dir", str(tmp_path / "s"), *common])
    run_cli([str(csv), str(b), "--out-dir", str(tmp_path / "p"), "--jobs", "2",
             *common])
    da, db = pd.read_csv(a), pd.read_csv(b)
    assert len(da) == len(db)
    for ra, rb in zip(da.itertuples(), db.itertuples()):
        x = sitk.GetArrayFromImage(sitk.ReadImage(ra.image_crop))
        y = sitk.GetArrayFromImage(sitk.ReadImage(rb.image_crop))
        assert np.array_equal(x, y)


def test_cropped_csv_trains(tmp_path):
    """The output CSV must be a drop-in for train.py."""
    csv = _cohort(tmp_path, n=6, with_empty=False)
    out_csv = tmp_path / "out.csv"
    run_cli([str(csv), str(out_csv), "--out-dir", str(tmp_path / "crops"),
             "--image-col", "ct", "--mask-col", "seg", "--for-crop-size", "48"])
    model = tmp_path / "m.pt"
    res = subprocess.run(
        [sys.executable, str(EXAMPLES / "train.py"), str(out_csv), "area_mm2",
         str(model), "--image-col", "image_crop", "--mask-col", "mask_crop",
         "--crop-size", "48", "--epochs", "2", "--warmup-epochs", "1",
         "--ramp-epochs", "0", "--batch-size", "2", "--val-frac", "0.34",
         "--device", "cpu"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert model.exists()


# --------------------------------------------------------------------------- #
# Field of view
# --------------------------------------------------------------------------- #
def test_required_patch_size_grows_with_the_structure():
    from ct25d.geometry import required_patch_size
    assert required_patch_size(10) < required_patch_size(40)
    assert required_patch_size(40) < required_patch_size(100)
    # 96 px at 0.78125 mm is a 75 mm field of view, so a 75 mm structure needs
    # considerably more than 96 px once the gate and augmentation are included
    assert required_patch_size(75) > 96
    assert required_patch_size(10) * 0.78125 > 10 + 2 * 10   # structure + gate


def test_mask_extent_is_physical(tmp_path):
    from ct25d.geometry import mask_extent_mm
    ip, mp = make_big_case(tmp_path, i=20, inplane=0.72, radius_px=10.0)
    ex, ey, ez = mask_extent_mm(sitk.ReadImage(str(mp)))
    assert ex == pytest.approx(2 * 10.0 * 0.72, abs=1.5)
    assert ey == pytest.approx(ex, abs=0.1)


def test_crop_reports_the_size_the_patch_needs(tmp_path):
    ip, mp = make_big_case(tmp_path, i=21, inplane=0.72, radius_px=30.0)
    stats = crop_pair(ip, mp, tmp_path / "f_img.nii.gz", tmp_path / "f_seg.nii.gz",
                      margin_mm=60.0)
    assert stats["extent_inplane_mm"] == pytest.approx(2 * 30.0 * 0.72, abs=1.5)
    assert stats["required_crop_size"] > 96          # 43 mm wide needs more


def test_train_refuses_a_patch_that_clips_the_structure(tmp_path):
    """The failure this guards against is silent: the mask just stops."""
    rows = []
    for i in range(4):
        ip, mp = make_big_case(tmp_path, i=30 + i, inplane=0.72,
                               label_at=(128, 128, 33), radius_px=45.0)
        rows.append({"ct": str(ip), "seg": str(mp), "area_mm2": 100.0 + i})
    csv = tmp_path / "big.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    args = [str(csv), "area_mm2", str(tmp_path / "m.pt"),
            "--image-col", "ct", "--mask-col", "seg", "--crop-size", "48",
            "--epochs", "1", "--batch-size", "2", "--val-frac", "0.5",
            "--device", "cpu"]
    res = subprocess.run([sys.executable, str(EXAMPLES / "train.py"), *args],
                         capture_output=True, text=True)
    assert res.returncode != 0
    combined = res.stdout + res.stderr
    assert "clipped" in combined and "--crop-size" in combined
    assert "--allow-clipped" in combined

    res = subprocess.run(
        [sys.executable, str(EXAMPLES / "train.py"), *args, "--allow-clipped"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "[warn]" in res.stdout


def test_train_is_quiet_when_everything_fits(tmp_path):
    rows = []
    for i in range(4):
        ip, mp = make_big_case(tmp_path, i=40 + i, radius_px=5.0)
        rows.append({"ct": str(ip), "seg": str(mp), "area_mm2": 100.0 + i})
    csv = tmp_path / "small.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = subprocess.run(
        [sys.executable, str(EXAMPLES / "train.py"), str(csv), "area_mm2",
         str(tmp_path / "m.pt"), "--image-col", "ct", "--mask-col", "seg",
         "--crop-size", "64", "--epochs", "1", "--batch-size", "2",
         "--val-frac", "0.5", "--device", "cpu"],
        capture_output=True, text=True)
    assert res.returncode == 0, res.stdout + res.stderr
    assert "every structure fits" in res.stdout
