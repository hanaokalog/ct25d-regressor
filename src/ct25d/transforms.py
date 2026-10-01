"""Geometric augmentation and target standardization."""

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["affine_theta", "RandomAffine2D", "rescale_target",
           "TargetStandardizer"]


def affine_theta(
    angle_deg,
    scale=1.0,
    shear_x=0.0,
    shear_y=0.0,
    translate_x=0.0,
    translate_y=0.0,
    height: int = 1,
    width: int = 1,
    hflip=None,
    device=None,
    dtype=torch.float32,
) -> torch.Tensor:
    """
    Build the (n, 2, 3) matrix F.affine_grid expects.

    The transform is composed in PIXEL space -- rotation therefore stays
    isotropic even when the image is not square -- then conjugated into the
    normalized [-1, 1] coordinates affine_grid works in, and finally inverted,
    because affine_grid maps output coordinates to input coordinates while the
    parameters here describe how the image content moves.

    translate_x / translate_y are fractions of the image width / height.
    Scalars and (n,) tensors are both accepted; separated from the sampling
    above so a transform can be built exactly for testing or for reproducing a
    specific augmentation.
    """
    def col(v):
        t = torch.as_tensor(v, device=device, dtype=dtype)
        return t.reshape(-1) if t.ndim else t.reshape(1)

    ang = col(angle_deg) * math.pi / 180.0
    s, shx, shy = col(scale), col(shear_x), col(shear_y)
    tx, ty = col(translate_x), col(translate_y)
    n = max(t.numel() for t in (ang, s, shx, shy, tx, ty))
    ang, s, shx, shy, tx, ty = (t.expand(n) for t in (ang, s, shx, shy, tx, ty))

    cos, sin = torch.cos(ang), torch.sin(ang)
    one = torch.ones_like(shx)
    R = torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)
    Sh = torch.stack([torch.stack([one, shx], -1), torch.stack([shy, one], -1)], -2)
    A_pix = s.view(n, 1, 1) * (R @ Sh)

    if hflip is not None:
        f = col(hflip).expand(n)
        A_pix = A_pix * torch.stack(
            [torch.stack([f, f], -1), torch.stack([one, one], -1)], -2)

    # pixel space -> normalized [-1, 1] space
    S = torch.diag(torch.tensor([2.0 / width, 2.0 / height],
                                device=device, dtype=dtype))
    A = S @ A_pix @ torch.inverse(S)

    M = torch.zeros(n, 3, 3, device=device, dtype=dtype)
    M[:, :2, :2] = A
    M[:, 0, 2] = 2.0 * tx          # normalized coordinates span 2 units
    M[:, 1, 2] = 2.0 * ty
    M[:, 2, 2] = 1.0
    return torch.inverse(M)[:, :2]


class RandomAffine2D:
    """
    One transform per SAMPLE, shared by every slice and the mask. Drawing an
    independent transform per slice would break the z correspondence that makes
    the neighbouring slices informative in the first place.

    Returns the sampled scale factor so a size-dependent label can be corrected.
    """

    def __init__(
        self,
        translate: float = 0.08,
        scale: tuple[float, float] = (0.9, 1.1),
        rotate_deg: float = 5.0,
        shear: float = 0.03,
        hflip: bool = False,
        z_flip: bool = False,               # reverse the slice order
        image_channels: Sequence[int] = (0, 1, 2),
        mask_channels: Sequence[int] = (3,),
        mask_threshold: Optional[float] = 0.5,
        image_padding: str = "border",      # 'border' avoids fake air outside CT
    ):
        self.translate, self.scale = translate, scale
        self.rotate_deg, self.shear = rotate_deg, shear
        self.hflip, self.z_flip = hflip, z_flip
        self.image_channels = tuple(image_channels)
        self.mask_channels = tuple(mask_channels)
        self.mask_threshold = mask_threshold
        self.image_padding = image_padding

    def _sample_theta(self, n, H, W, device, dtype):
        def u(lo, hi):
            return torch.empty(n, device=device, dtype=dtype).uniform_(lo, hi)

        s = u(*self.scale)
        flip = None
        if self.hflip:
            flip = torch.where(torch.rand(n, device=device) < 0.5,
                               -torch.ones(n, device=device, dtype=dtype),
                               torch.ones(n, device=device, dtype=dtype))
        theta = affine_theta(
            angle_deg=u(-self.rotate_deg, self.rotate_deg),
            scale=s,
            shear_x=u(-self.shear, self.shear),
            shear_y=u(-self.shear, self.shear),
            translate_x=u(-self.translate, self.translate),
            translate_y=u(-self.translate, self.translate),
            height=H, width=W, hflip=flip, device=device, dtype=dtype,
        )
        return theta, s

    def __call__(self, x: torch.Tensor):
        squeeze = x.dim() == 3
        if squeeze:
            x = x.unsqueeze(0)
        B, C, H, W = x.shape
        dtype = x.dtype if x.is_floating_point() else torch.float32
        theta, scale = self._sample_theta(B, H, W, x.device, dtype)
        grid = F.affine_grid(theta, (B, C, H, W), align_corners=False)

        out = torch.empty_like(x, dtype=dtype)
        img = list(self.image_channels)
        msk = list(self.mask_channels)
        if img:
            out[:, img] = F.grid_sample(x[:, img].to(dtype), grid, mode="bilinear",
                                        padding_mode=self.image_padding,
                                        align_corners=False)
        if msk:
            m = F.grid_sample(x[:, msk].to(dtype), grid, mode="bilinear",
                              padding_mode="zeros", align_corners=False)
            if self.mask_threshold is not None:
                m = (m >= self.mask_threshold).to(dtype)
            out[:, msk] = m

        if self.z_flip and img:
            flip = torch.rand(B, device=x.device) < 0.5
            if flip.any():
                sel = out[flip]                      # advanced indexing -> copy
                sel[:, img] = sel[:, img[::-1]]      # reverse the slice order
                out[flip] = sel

        if squeeze:
            out, scale = out[0], scale[0]
        return out, scale


def rescale_target(y, scale, power: int):
    """power = 0 scale-invariant, 1 length-like, 2 area-like."""
    return y * (scale ** power)


class TargetStandardizer:
    """z = (y - mean) / std, optionally on log1p(y). Fit on the training split."""

    def __init__(self, log_transform: bool = False, eps: float = 1e-8):
        self.log_transform, self.eps = log_transform, eps
        self.mean_: Optional[float] = None
        self.std_: Optional[float] = None

    def fit(self, y):
        y = np.asarray(y, dtype=np.float64).ravel()
        if self.log_transform:
            if np.any(y < 0):
                raise ValueError("log_transform requires non-negative targets")
            y = np.log1p(y)
        self.mean_, self.std_ = float(y.mean()), float(y.std() + self.eps)
        return self

    def transform(self, y):
        self._check()
        is_t = torch.is_tensor(y)
        z = y.double() if is_t else np.asarray(y, dtype=np.float64)
        if self.log_transform:
            z = torch.log1p(z) if is_t else np.log1p(z)
        z = (z - self.mean_) / self.std_
        return z.float() if is_t else z.astype(np.float32)

    def inverse_transform(self, mu, sigma=None):
        self._check()
        y = mu * self.std_ + self.mean_
        s = None if sigma is None else sigma * self.std_
        if self.log_transform:
            y_lin = torch.expm1(y) if torch.is_tensor(y) else np.expm1(y)
            s = None if s is None else s * (y_lin + 1)   # delta method
            y = y_lin
        return (y, s) if s is not None else y

    def interval(self, mu, sigma, z: float = 1.96):
        """
        (lo, hi) of the central interval mu +- z * sigma, built in the
        standardized space the Gaussian lives in and mapped back.

        With log_transform the predictive distribution of 1 + y is log-normal,
        so a symmetric mean +- 1.96 * sigma in the target's units (with the
        delta-method sigma) has the wrong coverage and its lower end can fall
        far below zero. The transform is monotonic, so mapping the two bounds
        keeps the probability exactly, and the lower bound stays above -1 (the
        log1p offset). Without log_transform this is the usual symmetric
        interval.
        """
        lo = self.inverse_transform(mu - z * sigma)
        hi = self.inverse_transform(mu + z * sigma)
        return lo, hi

    def state_dict(self):
        return {"mean": self.mean_, "std": self.std_, "log": self.log_transform}

    def load_state_dict(self, d):
        self.mean_, self.std_, self.log_transform = d["mean"], d["std"], d["log"]
        return self

    def _check(self):
        if self.mean_ is None:
            raise RuntimeError("TargetStandardizer is not fitted")
