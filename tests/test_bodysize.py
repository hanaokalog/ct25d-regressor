"""ct25d.bodysize on a phantom whose geometry is known exactly."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
torch = pytest.importorskip("torch")

from ct25d.bodysize import (  # noqa: E402
    FOV_RADII_MM,
    BodySizeDataset,
    BodySizeGeometry,
    BodySizeNet,
    level_z_mm,
    load_bodysize,
    load_case,
    predict_bodysize,
    prepare_case,
    save_bodysize,
    save_case,
)
from ct25d.transforms import TargetStandardizer  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
SMALL = BodySizeGeometry(proj_mm=4.0, rows=64, cols=64, l3_row=24,
                         axial_mm=4.0, axial_size=64)


def phantom(radius_mm=100.0, n_z=40, dz=5.0, px=2.0, n_xy=128, fov_mm=None,
            flip=False, z0=-100.0):
    """An elliptic-ish cylinder of 40 HU inside air, with a FOV circle of -3024."""
    yy, xx = (np.mgrid[0:n_xy, 0:n_xy] - (n_xy - 1) / 2.0) * px
    body = (xx / radius_mm) ** 2 + (yy / (0.7 * radius_mm)) ** 2 <= 1.0
    sl = np.where(body, 40.0, -1000.0).astype(np.float32)
    if fov_mm is not None:
        sl[np.hypot(xx, yy) > fov_mm / 2] = -3024.0
    vol = np.repeat(sl[None], n_z, 0)
    img = sitk.GetImageFromArray(vol)
    img.SetSpacing((px, px, dz))
    img.SetOrigin((-(n_xy - 1) / 2.0 * px, -(n_xy - 1) / 2.0 * px, z0))
    if flip:      # same anatomy stored with the z axis reversed
        img = sitk.GetImageFromArray(vol[::-1].copy())
        img.SetSpacing((px, px, dz))
        img.SetDirection((1, 0, 0, 0, 1, 0, 0, 0, -1))
        img.SetOrigin((-(n_xy - 1) / 2.0 * px, -(n_xy - 1) / 2.0 * px,
                       z0 + (n_z - 1) * dz))
    mask = sitk.GetArrayFromImage(img) > -500
    return img, mask


def test_level_z_mm_is_the_largest_slice():
    lab = np.zeros((10, 8, 8), np.uint8)
    lab[3, 2:4, 2:4] = 1
    lab[6, 1:7, 1:7] = 1
    im = sitk.GetImageFromArray(lab)
    im.SetSpacing((1.0, 1.0, 5.0))
    im.SetOrigin((0.0, 0.0, 10.0))
    assert level_z_mm(im) == pytest.approx(10.0 + 6 * 5.0)


def test_rows_widths_and_validity():
    img, mask = phantom()
    z3, z1 = -10.0, 70.0                  # volume spans z -100 .. 95
    c = prepare_case(img, mask, z1, z3, geometry=SMALL)
    assert c["proj"].shape == (len(FOV_RADII_MM), 6, 64, 64)
    assert c["axial"].shape == (2, 3, 2, 64, 64)
    assert c["l3_row"] == 24 and c["l1_row"] == pytest.approx(24 - 80 / 4.0)
    p = c["proj"][0].astype(np.float32)
    rows_scanned = np.flatnonzero(p[2].max(1) > 0)
    # row r is z = z3 + (24 - r) * 4: scanned rows are those inside -100 .. 95
    z = z3 + (24 - rows_scanned) * 4.0
    assert z.max() <= 95 + 2 and z.min() >= -100 - 2
    assert len(rows_scanned) == pytest.approx(195 / 4.0, abs=2)
    # frontal MIP (along y) is as wide as the body in x: 200 mm = 50 px
    width = (p[3, 24] > 0.5).sum()
    assert width == pytest.approx(200 / 4.0, abs=2)
    lateral = (p[0, 24] > 0.5).sum()      # along x: 140 mm deep
    assert lateral == pytest.approx(140 / 4.0, abs=2)
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_smaller_field_of_view_cuts_the_projection():
    img, mask = phantom()
    c = prepare_case(img, mask, 70.0, -10.0, geometry=SMALL)
    full = (c["proj"][0, 3, 24] > 0.5).sum()
    k = FOV_RADII_MM.index(175.0)
    assert (c["proj"][k, 3, 24] > 0.5).sum() == full     # 200 mm body fits a 350 mm FOV
    img2, mask2 = phantom(fov_mm=160.0)                  # a scanner that cuts it
    c2 = prepare_case(img2, mask2, 70.0, -10.0, geometry=SMALL)
    assert (c2["proj"][0, 3, 24] > 0.5).sum() == pytest.approx(160 / 4.0, abs=2)
    assert c2["fov_radius_mm"] == pytest.approx(128.0)   # half of 128 px * 2 mm


def test_the_stored_orientation_does_not_matter():
    a = prepare_case(*phantom(), 70.0, -10.0, geometry=SMALL)
    b = prepare_case(*phantom(flip=True), 70.0, -10.0, geometry=SMALL)
    assert np.allclose(a["proj"].astype(np.float32), b["proj"].astype(np.float32),
                       atol=1e-2)
    assert np.allclose(a["axial"][:, 1].astype(np.float32),
                       b["axial"][:, 1].astype(np.float32), atol=1e-2)


def test_augmentation_keeps_l1_to_l3_and_changes_per_epoch(tmp_path):
    c = prepare_case(*phantom(), 40.0, -10.0, geometry=SMALL)    # L1 on row 11.5
    save_case(tmp_path / "c.npz", c, SMALL)
    c = load_case(tmp_path / "c.npz")
    ds = BodySizeDataset([c] * 4, np.zeros((4, 2)), augment=True, p_trim=1.0)
    draws = set()
    for epoch in range(6):
        ds.set_epoch(epoch)
        proj, axial, _ = ds[0]
        valid_rows = np.flatnonzero(proj[2].numpy().max(1) > 0)
        assert valid_rows.min() <= c["l1_row"] - 5
        assert valid_rows.max() >= c["l3_row"] + 5
        draws.add((int(valid_rows.min()), int(valid_rows.max())))
        assert axial.shape == (4, 64, 64)
    assert len(draws) > 1


def test_model_checkpoint_and_prediction(tmp_path):
    c = prepare_case(*phantom(), 70.0, -10.0, geometry=SMALL)
    net = BodySizeNet()
    stds = {"height": TargetStandardizer().fit([150.0, 170.0]),
            "weight": TargetStandardizer().fit([50.0, 70.0])}
    cfg = dict(model=dict(norm="group", dropout=0.2, use_cbam=True, n_targets=2),
               targets=["height", "weight"], geometry=SMALL.as_dict())
    save_bodysize(tmp_path / "m.pt", net, stds, cfg, {"height": 1.0, "weight": 1.0})
    net2, stds2, cfg2, scales, _ = load_bodysize(tmp_path / "m.pt")
    out = predict_bodysize(net2, stds2, scales, [c, c], cfg2["targets"])
    mean, sigma, lo, hi = out["height"]
    assert mean.shape == (2,) and np.all(sigma > 0) and np.all(lo < mean)
    assert np.all(mean < hi)
    assert np.allclose(mean[0], mean[1])


def test_train_bodysize_cli(tmp_path):
    import pandas as pd
    rows = []
    for i in range(10):
        r = 70.0 + 6 * i
        c = prepare_case(*phantom(radius_mm=r), 70.0, -10.0, geometry=SMALL)
        p = tmp_path / f"c{i}.npz"
        save_case(p, c, SMALL)
        split = (["train"] * 6 + ["val"] * 2 + ["test"] * 2)[i]
        rows.append(dict(case=str(p), height=150 + i, weight=r * 0.6, split=split))
    rows[0]["weight"] = np.nan                         # left out of training
    pd.DataFrame(rows).to_csv(tmp_path / "cases.csv", index=False)
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / "train_bodysize.py"),
         str(tmp_path / "cases.csv"), str(tmp_path / "bs.pt"),
         "--split-col", "split", "--epochs", "2",
         "--warmup-epochs", "1", "--ramp-epochs", "0", "--batch-size", "2",
         "--device", "cpu"], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "9 cases with every target" in out.stdout
    assert "test (held out" in out.stdout
    _, _, cfg, scales, metrics = load_bodysize(tmp_path / "bs.pt")
    assert cfg["geometry"]["rows"] == 64 and set(scales) == {"height", "weight"}
    assert "test_weight_mae" in metrics
