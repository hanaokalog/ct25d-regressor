"""2.5D ResNet + CBAM backbone with a heteroscedastic regression head, or a
classification head when `n_classes` is given."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:                       # avoids a runtime import cycle
    from .transforms import TargetStandardizer

__all__ = ["make_norm", "CBAM2D", "BasicBlockCBAM2D", "ResNet25DCBAMRegressor",
           "resnet18_cbam25d", "resnet34_cbam25d"]


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "group":
        groups = next(g for g in (8, 4, 2, 1) if channels % g == 0)
        return nn.GroupNorm(groups, channels)
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm: {kind}")


class ChannelAttention2D(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
        )

    def forward(self, x):
        avg = self.mlp(F.adaptive_avg_pool2d(x, 1))
        mx = self.mlp(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class SpatialAttention2D(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size % 2 == 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx = x.amax(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM2D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttention2D(channels, reduction)
        self.spatial = SpatialAttention2D(kernel_size)

    def forward(self, x):
        return self.spatial(self.channel(x))


def conv3x3(cin, cout, stride=1):
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


class BasicBlockCBAM2D(nn.Module):
    expansion = 1

    def __init__(self, in_ch, out_ch, stride=1, norm="batch", reduction=16,
                 sa_kernel=7, use_cbam=True):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride)
        self.norm1 = make_norm(norm, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.norm2 = make_norm(norm, out_ch)
        self.cbam = CBAM2D(out_ch, reduction, sa_kernel) if use_cbam else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                (nn.AvgPool2d(stride, stride, ceil_mode=True)
                 if stride != 1 else nn.Identity()),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                make_norm(norm, out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.cbam(out)
        return self.relu(out + self.shortcut(x))


class ResNet25DCBAMRegressor(nn.Module):
    def __init__(
        self,
        n_slices: int = 3,
        n_mask_channels: int = 1,
        block: type = BasicBlockCBAM2D,
        layers: Sequence[int] = (2, 2, 2, 2),
        widths: Sequence[int] = (32, 64, 128, 256),
        stem_stride: int = 2,
        stem_pool: bool = True,
        norm: str = "batch",
        reduction: int = 16,
        sa_kernels: Sequence[int] = (7, 7, 3, 3),
        use_cbam: bool = True,
        dropout: float = 0.2,
        logvar_min: float = -7.0,
        logvar_max: float = 7.0,
        logvar_margin: float = 1.0,
        n_classes: int = 0,
    ):
        super().__init__()
        assert len(layers) == len(widths) == len(sa_kernels)
        in_channels = n_slices + n_mask_channels
        self.n_slices = n_slices
        self.logvar_min, self.logvar_max = logvar_min, logvar_max
        self.logvar_margin = float(logvar_margin)

        stem_ch = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_ch, 7, stride=stem_stride,
                      padding=3, bias=False),
            make_norm(norm, stem_ch), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(3, 2, padding=1) if stem_pool else nn.Identity()

        stages, in_ch = [], stem_ch
        for i, (n, w, k) in enumerate(zip(layers, widths, sa_kernels)):
            blocks = []
            for j in range(n):
                stride = 2 if (j == 0 and i > 0) else 1
                blocks.append(block(in_ch, w, stride, norm, reduction, k, use_cbam))
                in_ch = w * block.expansion
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.out_channels = in_ch

        self.dropout = nn.Dropout(dropout)
        # n_classes > 0 swaps the Gaussian head for class logits; the backbone
        # is the same, so everything upstream of the head is shared.
        self.n_classes = int(n_classes)
        if self.n_classes > 0:
            self.fc_logits = nn.Linear(in_ch, self.n_classes)
        else:
            self.fc_mu = nn.Linear(in_ch, 1)
            self.fc_logvar = nn.Linear(in_ch, 1)
        self._init_weights()

    @property
    def is_classifier(self) -> bool:
        return self.n_classes > 0

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, BasicBlockCBAM2D) and hasattr(m.norm2, "weight"):
                nn.init.zeros_(m.norm2.weight)
        if self.is_classifier:
            nn.init.zeros_(self.fc_logits.bias)
            return
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.bias)
        nn.init.normal_(self.fc_logvar.weight, std=1e-3)

    def _bound_logvar(self, v: torch.Tensor) -> torch.Tensor:
        """
        Two-sided bound on the log-variance: exactly the identity in the
        interior, smoothly saturating within `logvar_margin` of each end.

        A hard clamp zeroes the gradient the moment the head saturates, so a
        variance head that overshoots can never come back. A plain scaled tanh
        fixes that but compresses the whole range -- with +-7 bounds it turns
        log_var = 2 into 1.95, a 2.6% error in sigma exactly where the model is
        reporting high uncertainty. Bending only the last `logvar_margin` units
        keeps the working range exact and still leaves a non-zero gradient
        outside it, since tanh saturates asymptotically rather than abruptly.
        """
        m = self.logvar_margin
        hi, lo = self.logvar_max - m, self.logvar_min + m
        v = torch.where(v > hi, hi + m * torch.tanh((v - hi) / m), v)
        return torch.where(v < lo, lo - m * torch.tanh((lo - v) / m), v)

    def forward(self, x):
        x = self.pool(self.stem(x))
        for stage in self.stages:
            x = stage(x)
        x = self.dropout(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))
        if self.is_classifier:
            return self.fc_logits(x)
        return self.fc_mu(x), self._bound_logvar(self.fc_logvar(x))

    @torch.no_grad()
    def predict_proba(self, x, temperature: float = 1.0):
        """Class probabilities, with the logits divided by `temperature`."""
        if not self.is_classifier:
            raise RuntimeError("predict_proba needs a model built with n_classes")
        self.eval()
        return torch.softmax(self(x) / float(temperature), dim=1)

    @torch.no_grad()
    def predict(self, x, standardizer: Optional["TargetStandardizer"] = None):
        if self.is_classifier:
            raise RuntimeError("a classifier predicts with predict_proba")
        self.eval()
        mu, log_var = self(x)
        sigma = torch.exp(0.5 * log_var)
        if standardizer is None:
            return mu, sigma
        return standardizer.inverse_transform(mu, sigma)


def resnet18_cbam25d(**kw):
    return ResNet25DCBAMRegressor(layers=(2, 2, 2, 2), widths=(32, 64, 128, 256), **kw)


def resnet34_cbam25d(**kw):
    return ResNet25DCBAMRegressor(layers=(3, 4, 6, 3), widths=(32, 64, 128, 256), **kw)
