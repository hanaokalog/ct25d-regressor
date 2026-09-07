"""Backbone: shapes, gradients, and the heteroscedastic head."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ct25d.models import CBAM2D, resnet18_cbam25d
from ct25d.transforms import TargetStandardizer


def test_forward_shapes():
    m = resnet18_cbam25d(n_slices=3, n_mask_channels=1, norm="group")
    mu, log_var = m(torch.randn(2, 4, 96, 96))
    assert mu.shape == (2, 1) and log_var.shape == (2, 1)


@pytest.mark.parametrize("n_slices,n_mask", [(3, 1), (3, 2), (1, 1), (5, 1)])
def test_channel_configurations(n_slices, n_mask):
    m = resnet18_cbam25d(n_slices=n_slices, n_mask_channels=n_mask, norm="group")
    mu, _ = m(torch.randn(2, n_slices + n_mask, 64, 64))
    assert mu.shape == (2, 1)


@pytest.mark.parametrize("size", [64, 96, 128, 160])
def test_variable_input_size(size):
    m = resnet18_cbam25d(norm="group")
    mu, _ = m(torch.randn(1, 4, size, size))
    assert mu.shape == (1, 1)


def test_non_square_input():
    m = resnet18_cbam25d(norm="group")
    mu, _ = m(torch.randn(1, 4, 96, 128))
    assert mu.shape == (1, 1)


def test_residual_branches_are_silent_at_initialization_but_unfreeze():
    """
    Zero-initialized gamma makes every block an exact identity at step 0, which
    means the layers inside the branch get no gradient on the first backward --
    expected, and the reason this is checked rather than asserted away. The
    gamma itself does get a gradient, so one optimizer step unfreezes the rest.
    """
    torch.manual_seed(0)                      # keep the check deterministic
    m = resnet18_cbam25d(norm="group")
    opt = torch.optim.SGD(m.parameters(), lr=0.05)

    def step_loss():
        mu, log_var = m(torch.randn(4, 4, 64, 64))
        return ((mu - 1.0) ** 2).mean() + (log_var ** 2).mean()

    step_loss().backward()
    gammas = [p for n, p in m.named_parameters() if n.endswith("norm2.weight")]
    assert all(float(g.grad.abs().sum()) > 0 for g in gammas)

    for _ in range(3):
        opt.step()
        opt.zero_grad(set_to_none=True)
        step_loss().backward()
    missing = [n for n, p in m.named_parameters()
               if p.requires_grad and (p.grad is None or torch.all(p.grad == 0))]
    assert missing == [], missing


@pytest.mark.parametrize("bias", [50.0, -50.0])
def test_log_var_is_bounded(bias):
    m = resnet18_cbam25d(norm="group", logvar_min=-2.0, logvar_max=2.0)
    with torch.no_grad():
        m.fc_logvar.bias.fill_(bias)
    _, log_var = m(torch.randn(2, 4, 64, 64))
    assert -2.0 <= float(log_var.min().detach())
    assert float(log_var.max().detach()) <= 2.0


@pytest.mark.parametrize("bias", [3.0, 6.0, -6.0])
def test_saturated_log_var_can_still_recover(bias):
    """A hard clamp would zero this gradient and strand the variance head."""
    m = resnet18_cbam25d(norm="group", logvar_min=-2.0, logvar_max=2.0)
    with torch.no_grad():
        m.fc_logvar.bias.fill_(bias)          # well outside the bound
    _, log_var = m(torch.randn(2, 4, 64, 64))
    log_var.sum().backward()
    assert float(m.fc_logvar.bias.grad.abs().sum()) > 0


def test_bound_is_near_identity_in_the_working_range():
    m = resnet18_cbam25d(norm="group")        # default bounds +-7
    v = torch.tensor([[-6.0], [-2.0], [-0.5], [0.0], [0.5], [2.0], [6.0]])
    out = m._bound_logvar(v)
    assert torch.allclose(out, v, atol=1e-6)     # exact, not merely close


def test_initial_sigma_is_about_one():
    """A variance head that starts at sigma != 1 makes the NLL warmup pointless."""
    m = resnet18_cbam25d(norm="group")
    m.eval()
    _, log_var = m(torch.randn(8, 4, 64, 64))
    sigma = float(torch.exp(0.5 * log_var).mean().detach())
    assert sigma == pytest.approx(1.0, abs=0.05)


def test_residual_blocks_start_as_identity():
    m = resnet18_cbam25d(norm="group")
    for name, p in m.named_parameters():
        if name.endswith("norm2.weight"):
            assert float(p.detach().abs().max()) == 0.0


@pytest.mark.parametrize("norm", ["group", "batch", "instance", "none"])
def test_norm_variants(norm):
    m = resnet18_cbam25d(norm=norm)
    m.train()
    mu, _ = m(torch.randn(4, 4, 64, 64))
    assert torch.isfinite(mu).all()


def test_group_norm_works_with_batch_size_one():
    m = resnet18_cbam25d(norm="group").train()
    assert torch.isfinite(m(torch.randn(1, 4, 64, 64))[0]).all()


def test_cbam_preserves_shape_and_bounds():
    x = torch.randn(2, 32, 16, 16)
    out = CBAM2D(32)(x)
    assert out.shape == x.shape
    ratio = (out / (x + 1e-9)).abs()
    assert float(ratio.detach().max()) <= 1.0 + 1e-4       # attention only attenuates


def test_disabling_cbam_reduces_the_parameter_count():
    a = sum(p.numel() for p in resnet18_cbam25d(use_cbam=True).parameters())
    b = sum(p.numel() for p in resnet18_cbam25d(use_cbam=False).parameters())
    assert b < a


def test_predict_returns_physical_units():
    m = resnet18_cbam25d(norm="group")
    std = TargetStandardizer().fit(np.array([100.0, 200.0, 300.0, 400.0]))
    mean, sigma = m.predict(torch.randn(3, 4, 64, 64), std)
    assert mean.shape == (3, 1) and sigma.shape == (3, 1)
    assert float(sigma.min()) > 0
    assert 0 < float(mean.abs().max()) < 10000


def test_deterministic_in_eval_mode():
    m = resnet18_cbam25d(norm="group", dropout=0.5).eval()
    x = torch.randn(2, 4, 64, 64)
    with torch.no_grad():
        assert torch.allclose(m(x)[0], m(x)[0])
