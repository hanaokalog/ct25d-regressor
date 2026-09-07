"""End-to-end test of the two command line scripts.

Writes real .nii.gz files with deliberately different voxel and matrix sizes,
runs train.py, then runs eval.py against the checkpoint it produced. Slow
compared to the unit tests, but it is the only check that the CSV contract, the
checkpoint round trip and the two scripts actually fit together.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")
torch = pytest.importorskip("torch")
pd = pytest.importorskip("pandas")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def write_case(directory: Path, i: int, rng):
    """A phantom whose disc area is the regression target, in mm^2."""
    # deliberately heterogeneous acquisitions
    inplane = float(rng.choice([0.68, 0.78125, 0.98]))
    z_spacing = float(rng.choice([1.25, 2.5, 5.0]))
    W = H = int(rng.choice([96, 128]))
    Z = 10
    cz = Z // 2
    r_px = rng.uniform(4.0, 12.0)

    yy, xx = np.mgrid[0:H, 0:W]
    disc = ((yy - H / 2) ** 2 + (xx - W / 2) ** 2 <= r_px ** 2)
    vol = rng.normal(-50, 20, (Z, H, W)).astype(np.float32)
    for z in range(Z):
        shrink = max(0.0, 1.0 - abs(z - cz) / 6.0)
        vol[z] += 800.0 * (((yy - H / 2) ** 2 + (xx - W / 2) ** 2)
                           <= (r_px * shrink) ** 2)
    mask = np.zeros((Z, H, W), np.uint8)
    mask[cz] = disc

    img, lab = sitk.GetImageFromArray(vol), sitk.GetImageFromArray(mask)
    for it in (img, lab):
        it.SetSpacing((inplane, inplane, z_spacing))
        it.SetOrigin((-10.0 * i, 5.0 * i, 0.0))

    ipath = directory / f"case{i:03d}_img.nii.gz"
    mpath = directory / f"case{i:03d}_seg.nii.gz"
    sitk.WriteImage(img, str(ipath))
    sitk.WriteImage(lab, str(mpath))
    area = float(disc.sum()) * inplane * inplane
    return {"ct": str(ipath), "seg": str(mpath), "patient_id": f"p{i // 2:03d}",
            "area_mm2": area, "unrelated": rng.normal()}


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    d = tmp_path_factory.mktemp("cohort")
    rng = np.random.default_rng(0)
    rows = [write_case(d, i, rng) for i in range(12)]
    csv = d / "cases.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def run(script, argv):
    """Run a CLI in a subprocess, the way a user would."""
    out = subprocess.run([sys.executable, str(EXAMPLES / script), *argv],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    return out.stdout


def test_train_then_eval(cohort, tmp_path):
    model = tmp_path / "model.pt"
    log = run("train.py", [
        str(cohort), "area_mm2", str(model),
        "--image-col", "ct", "--mask-col", "seg", "--group-col", "patient_id",
        "--crop-size", "48", "--target-power", "2", "--log-target",
        "--epochs", "4", "--warmup-epochs", "1", "--ramp-epochs", "1",
        "--batch-size", "4", "--val-frac", "0.34", "--device", "cpu",
    ])
    assert model.exists()
    assert "wrote" in log and "sigma scale" in log

    from ct25d.checkpoint import load_checkpoint
    net, std, cfg, scale, metrics = load_checkpoint(model)
    assert cfg["target"] == "area_mm2"
    assert cfg["crop_size"] == 48 and cfg["log_target"] is True
    assert cfg["target_power"] == 2
    assert std.mean_ is not None and scale > 0
    assert "mae" in metrics

    pred = tmp_path / "pred.csv"
    log = run("eval.py", [str(cohort), "area_mm2", str(model),
                          "--out", str(pred), "--device", "cpu"])
    assert pred.exists()
    out = pd.read_csv(pred)
    assert len(out) == 12
    for col in ("pred", "pred_sigma", "pred_lo95", "pred_hi95",
                "residual", "z_score"):
        assert col in out.columns
    assert (out["pred_sigma"] > 0).all()
    assert (out["pred_hi95"] > out["pred_lo95"]).all()
    assert np.isfinite(out["pred"]).all()
    assert "unrelated" in out.columns          # the original columns survive


def test_epoch_log_reports_the_error_in_physical_units(cohort, tmp_path):
    """A standardized MAE cannot be compared with clinical expectations."""
    log = run("train.py", [
        str(cohort), "area_mm2", str(tmp_path / "m.pt"),
        "--image-col", "ct", "--mask-col", "seg", "--crop-size", "48",
        "--epochs", "3", "--warmup-epochs", "1", "--ramp-epochs", "1",
        "--batch-size", "4", "--val-frac", "0.34", "--device", "cpu",
    ])
    assert "val_mae is in units of area_mm2" in log
    epochs = [ln for ln in log.splitlines() if ln.startswith("epoch ")]
    assert len(epochs) == 3
    for ln in epochs:
        assert "area_mm2" in ln and "sd)" in ln
    # the physical value must differ from the standardized one, and be on the
    # scale of the targets themselves
    val = float(epochs[0].split("val_mae")[1].split()[0])
    sd = float(epochs[0].split("(")[1].split()[0])
    assert val > 0 and val != pytest.approx(sd)
    truth = pd.read_csv(cohort)["area_mm2"]
    assert val < 10 * truth.max()


def test_eval_without_the_target_column(cohort, tmp_path):
    model = tmp_path / "model.pt"
    run("train.py", [str(cohort), "area_mm2", str(model),
                     "--image-col", "ct", "--mask-col", "seg",
                     "--crop-size", "48", "--epochs", "2", "--warmup-epochs", "1",
                     "--ramp-epochs", "0", "--batch-size", "4",
                     "--val-frac", "0.34", "--device", "cpu"])

    unlabelled = tmp_path / "unlabelled.csv"
    df = pd.read_csv(cohort).drop(columns=["area_mm2"])
    df.to_csv(unlabelled, index=False)

    pred = tmp_path / "pred2.csv"
    log = run("eval.py", [str(unlabelled), "area_mm2", str(model),
                          "--out", str(pred), "--device", "cpu"])
    assert "predicting only" in log
    out = pd.read_csv(pred)
    assert "pred" in out.columns and "residual" not in out.columns


def test_train_rejects_an_unknown_target(cohort, tmp_path):
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / "train.py"), str(cohort), "nope",
         str(tmp_path / "m.pt"), "--image-col", "ct", "--mask-col", "seg"],
        capture_output=True, text=True)
    assert out.returncode != 0
    assert "not in the CSV" in out.stdout + out.stderr
