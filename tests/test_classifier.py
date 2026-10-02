"""Classification head, temperature scaling and the HU shift augmentation."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ct25d.calibration import classification_report, fit_temperature
from ct25d.checkpoint import build_model, load_checkpoint, save_checkpoint
from ct25d.data import SliceStackDataset


def test_classifier_outputs_logits_and_round_trips(tmp_path):
    arch = dict(name="resnet18", n_slices=3, n_mask_channels=1, norm="group",
                n_classes=5)
    net = build_model(arch)
    x = torch.randn(2, 4, 64, 64)
    logits = net.eval()(x)
    assert logits.shape == (2, 5)
    p = net.predict_proba(x, temperature=2.0)
    assert torch.allclose(p.sum(1), torch.ones(2), atol=1e-6)
    with pytest.raises(RuntimeError):
        net.predict(x)

    path = tmp_path / "c.pt"
    save_checkpoint(path, net, None, {"arch": arch, "task": "classification"},
                    temperature=1.7)
    net2, std, cfg, _, _ = load_checkpoint(path)
    assert std is None and cfg["temperature"] == pytest.approx(1.7)
    assert torch.allclose(net2(x), logits, atol=1e-5)


def test_temperature_undoes_an_overconfident_scale():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 4, 4000)
    base = rng.normal(size=(4000, 4))
    base[np.arange(4000), y] += 1.0          # informative but noisy logits
    t = fit_temperature(base * 3.0, y)       # same model, 3x overconfident
    t0 = fit_temperature(base, y)
    assert t == pytest.approx(3.0 * t0, rel=0.02)


def test_classification_report():
    p = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
    rep = classification_report(p, [0, 1, 1])
    assert rep["accuracy"] == pytest.approx(2 / 3)
    assert rep["confusion"] == [[1, 0], [1, 1]]
    assert 0 <= rep["ece"] <= 1


def test_hu_shift_is_uniform_over_the_image_and_spares_the_mask():
    stack = np.zeros((1, 4, 16, 16), np.float32)
    stack[0, :3] = 100.0
    stack[0, 3, 4:12, 4:12] = 1.0
    ds = SliceStackDataset(stack, [0], None, gate=None, mask_channel="binary",
                           window=(-1000.0, 1000.0), hu_shift=20.0,
                           task="classification")
    vals = set()
    for _ in range(20):
        x, y = ds[0]
        img = x[:3].numpy()
        assert np.allclose(img, img.flat[0])           # one offset per case
        hu = (img.flat[0] + 1) / 2 * 2000 - 1000
        assert 80 - 1e-3 <= hu <= 120 + 1e-3
        assert set(np.unique(x[3].numpy())) <= {0.0, 1.0}
        assert y.dtype == torch.long
        vals.add(round(float(hu), 3))
    assert len(vals) > 1


def test_format_1_checkpoints_still_load_and_unknown_ones_are_refused(tmp_path):
    from ct25d.transforms import TargetStandardizer

    arch = dict(name="resnet18", n_slices=3, n_mask_channels=1, norm="group")
    net = build_model(arch)
    std = TargetStandardizer().fit([1.0, 3.0])
    old = {"format_version": 1, "state_dict": net.state_dict(),
           "standardizer": std.state_dict(), "config": {"arch": arch},
           "sigma_scale": 1.5, "metrics": {}}      # no task, no temperature
    torch.save(old, tmp_path / "v1.pt")
    _, std2, cfg, scale, _ = load_checkpoint(tmp_path / "v1.pt")
    assert cfg["task"] == "regression" and cfg["temperature"] == 1.0
    assert scale == 1.5 and std2.mean_ == std.mean_
    torch.save(dict(old, format_version=99), tmp_path / "v99.pt")
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / "v99.pt")
