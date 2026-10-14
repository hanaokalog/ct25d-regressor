"""Gaussian negative log-likelihood, with a Huber warmup schedule; class weights for classification."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GaussianNLL", "WarmupHeteroscedasticLoss", "class_weights"]


def class_weights(spec: str, counts) -> np.ndarray | None:
    """Per-class loss weights from the training class counts.

    spec: "none" (None), "balanced" (inverse frequency), "sqrt" (square root of
    that, a milder push), or a comma list such as "1,1,1,1,2,1". A class with no
    training case gets 0. The weights are scaled so that the mean weight over the
    training cases is 1, which keeps the loss (and the learning rate) on the
    unweighted scale.
    """
    counts = np.asarray(counts, dtype=float)
    if spec in (None, "", "none"):
        return None
    if spec in ("balanced", "sqrt"):
        w = np.where(counts > 0, counts.sum() / (len(counts) * np.maximum(counts, 1)), 0.0)
        if spec == "sqrt":
            w = np.sqrt(w)
    else:
        w = np.array([float(v) for v in spec.split(",")])
        if len(w) != len(counts) or (w < 0).any():
            raise ValueError(f"class weights {spec!r}: need {len(counts)} non-negative values")
    return w * counts.sum() / max((w * counts).sum(), 1e-12)


class GaussianNLL(nn.Module):
    """beta > 0 enables beta-NLL (Seitzer et al., ICLR 2022)."""

    def __init__(self, beta: float = 0.5, full: bool = False):
        super().__init__()
        self.beta, self.full = beta, full

    def forward(self, mu, log_var, target):
        target = target.view_as(mu)
        var = torch.exp(log_var)
        nll = 0.5 * (log_var + (target - mu) ** 2 / var)
        if self.full:
            nll = nll + 0.5 * math.log(2 * math.pi)
        if self.beta > 0:
            nll = nll * var.detach() ** self.beta
        return nll.mean()


class WarmupHeteroscedasticLoss(nn.Module):
    """Huber for `warmup_epochs`, linearly blended into Gaussian NLL."""

    def __init__(self, warmup_epochs=5, ramp_epochs=5, beta=0.5,
                 huber_delta=1.0, anchor=1e-2):
        super().__init__()
        self.warmup_epochs, self.ramp_epochs = warmup_epochs, ramp_epochs
        self.huber_delta, self.anchor = huber_delta, anchor
        self.nll = GaussianNLL(beta=beta)

    def alpha(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        t = (epoch - self.warmup_epochs) / self.ramp_epochs
        return float(min(max(t, 0.0), 1.0))

    def forward(self, mu, log_var, target, epoch: int):
        target = target.view_as(mu)
        a = self.alpha(epoch)
        loss = torch.zeros((), device=mu.device, dtype=mu.dtype)
        if a < 1.0:
            huber = F.smooth_l1_loss(mu, target, beta=self.huber_delta)
            loss = loss + (1.0 - a) * huber
            if self.anchor > 0:
                loss = loss + (1.0 - a) * self.anchor * (log_var ** 2).mean()
        if a > 0.0:
            loss = loss + a * self.nll(mu, log_var, target)
        return loss
