"""Gaussian negative log-likelihood, with a Huber warmup schedule."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["GaussianNLL", "WarmupHeteroscedasticLoss"]


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
