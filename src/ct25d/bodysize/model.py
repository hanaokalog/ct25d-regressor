"""Two-branch network for height and weight, with a Gaussian head per target."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models import ResNet25DCBAMRegressor

__all__ = ["BodySizeNet", "TARGETS"]

TARGETS = ("height", "weight")

PROJ_CHANNELS = 6     # MIP, mean, validity along x and along y
AXIAL_CHANNELS = 4    # (value, validity) at L1 and L3


class _Backbone(nn.Module):
    """The ct25d ResNet-CBAM trunk without its head."""

    def __init__(self, in_channels, norm="group", widths=(32, 64, 128, 256),
                 layers=(2, 2, 2, 2), use_cbam=True):
        super().__init__()
        net = ResNet25DCBAMRegressor(n_slices=in_channels, n_mask_channels=0,
                                     norm=norm, widths=widths, layers=layers,
                                     use_cbam=use_cbam, dropout=0.0)
        self.stem, self.pool, self.stages = net.stem, net.pool, net.stages
        self.out_channels = net.out_channels
        self._bound = net._bound_logvar

    def forward(self, x):
        x = self.pool(self.stem(x))
        for stage in self.stages:
            x = stage(x)
        return torch.flatten(F.adaptive_avg_pool2d(x, 1), 1)


class BodySizeNet(nn.Module):
    """
    proj  (B, 6, rows, cols) and axial (B, 4, S, S) -> per target (mu, log_var)
    in the standardized space, as two (B, n_targets) tensors.
    """

    def __init__(self, norm="group", dropout=0.2, use_cbam=True, n_targets=2):
        super().__init__()
        self.proj = _Backbone(PROJ_CHANNELS, norm=norm, use_cbam=use_cbam)
        self.axial = _Backbone(AXIAL_CHANNELS, norm=norm, use_cbam=use_cbam)
        feat = self.proj.out_channels + self.axial.out_channels
        self.dropout = nn.Dropout(dropout)
        self.fc_mu = nn.Linear(feat, n_targets)
        self.fc_logvar = nn.Linear(feat, n_targets)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.bias)
        nn.init.normal_(self.fc_logvar.weight, std=1e-3)

    def forward(self, proj, axial):
        f = self.dropout(torch.cat([self.proj(proj), self.axial(axial)], 1))
        return self.fc_mu(f), self.proj._bound(self.fc_logvar(f))
