"""Gaussian NLL and the Huber -> NLL warmup schedule."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ct25d.losses import GaussianNLL, WarmupHeteroscedasticLoss


def test_nll_matches_the_closed_form():
    mu = torch.tensor([[1.0], [2.0]])
    log_var = torch.tensor([[0.0], [math.log(4.0)]])
    y = torch.tensor([[1.5], [0.0]])
    got = float(GaussianNLL(beta=0.0, full=True)(mu, log_var, y))
    want = np.mean([0.5 * (math.log(2 * math.pi * 1.0) + 0.25 / 1.0),
                    0.5 * (math.log(2 * math.pi * 4.0) + 4.0 / 4.0)])
    assert got == pytest.approx(want, rel=1e-6)


def test_nll_is_minimized_at_the_true_sigma():
    y = torch.randn(4096, 1) * 3.0
    mu = torch.zeros_like(y)
    def nll(s):
        return float(GaussianNLL(beta=0.0)(
            mu, torch.full_like(y, 2 * math.log(s)), y))

    losses = {s: nll(s) for s in (1.0, 2.0, 3.0, 4.5, 8.0)}
    assert min(losses, key=losses.get) == 3.0


def test_beta_nll_downweights_high_variance_terms():
    mu = torch.zeros(2, 1)
    y = torch.tensor([[1.0], [1.0]])
    log_var = torch.tensor([[0.0], [math.log(9.0)]])
    plain = GaussianNLL(beta=0.0)(mu, log_var, y)
    beta = GaussianNLL(beta=0.5)(mu, log_var, y)
    assert float(beta) != float(plain)


def test_warmup_alpha_schedule():
    c = WarmupHeteroscedasticLoss(warmup_epochs=5, ramp_epochs=5)
    assert [c.alpha(e) for e in (0, 4)] == [0.0, 0.0]
    assert c.alpha(5) == 0.0 and c.alpha(7) == pytest.approx(0.4)
    assert c.alpha(10) == 1.0 and c.alpha(99) == 1.0


def test_warmup_with_zero_ramp_switches_abruptly():
    c = WarmupHeteroscedasticLoss(warmup_epochs=3, ramp_epochs=0)
    assert c.alpha(2) == 0.0 and c.alpha(3) == 1.0


def test_during_warmup_the_loss_does_not_depend_on_sigma_except_the_anchor():
    c = WarmupHeteroscedasticLoss(warmup_epochs=5, anchor=0.0)
    mu, y = torch.zeros(4, 1), torch.ones(4, 1)
    a = float(c(mu, torch.zeros(4, 1), y, epoch=0))
    b = float(c(mu, torch.full((4, 1), 3.0), y, epoch=0))
    assert a == pytest.approx(b)


def test_anchor_keeps_the_variance_head_in_the_graph():
    """Without this, DDP without find_unused_parameters fails during warmup."""
    c = WarmupHeteroscedasticLoss(warmup_epochs=5, anchor=1e-2)
    log_var = torch.full((4, 1), 2.0, requires_grad=True)
    loss = c(torch.zeros(4, 1, requires_grad=True), log_var, torch.ones(4, 1), epoch=0)
    loss.backward()
    assert log_var.grad is not None and float(log_var.grad.abs().sum()) > 0


def test_after_the_ramp_the_loss_equals_the_plain_nll():
    c = WarmupHeteroscedasticLoss(warmup_epochs=1, ramp_epochs=0, beta=0.5)
    mu, log_var, y = torch.zeros(4, 1), torch.zeros(4, 1), torch.ones(4, 1)
    assert float(c(mu, log_var, y, epoch=9)) == pytest.approx(
        float(GaussianNLL(beta=0.5)(mu, log_var, y)))


def test_loss_is_finite_at_the_clamp_extremes():
    for lv in (-7.0, 7.0):
        loss = GaussianNLL()(torch.zeros(4, 1), torch.full((4, 1), lv),
                             torch.ones(4, 1))
        assert torch.isfinite(loss)
