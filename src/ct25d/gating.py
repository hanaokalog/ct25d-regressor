"""Distance-transform gating, windowing, and input assembly."""

from collections.abc import Sequence
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt

from .constants import CT_WINDOW, TARGET_INPLANE_MM

__all__ = ["falloff", "DistanceGate", "window_01", "gate_image",
           "signed_distance_channel", "make_input"]


def falloff(t: np.ndarray, profile: str = "cosine") -> np.ndarray:
    """
    t is the normalized distance d / R, already clipped to [0, 1].
    Returns the weight, 1 at t=0 and 0 at t=1.

    linear     : 1 - t                      slope -1/R at both ends
    cosine     : (1 + cos(pi t)) / 2        slope 0 at both ends  <- default
    smoothstep : 1 - (3t^2 - 2t^3)          slope 0 at both ends, slightly wider
                                            plateau near the mask
    gaussian   : exp(-t^2 / (2 s^2)) truncated and rescaled to hit 0 at t=1
    """
    if profile == "linear":
        return 1.0 - t
    if profile == "cosine":
        return 0.5 * (1.0 + np.cos(np.pi * t))
    if profile == "smoothstep":
        return 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)
    if profile == "gaussian":
        s = 0.35
        g = np.exp(-(t ** 2) / (2 * s ** 2))
        g0, g1 = 1.0, float(np.exp(-1.0 / (2 * s ** 2)))
        return (g - g1) / (g0 - g1)
    raise ValueError(f"unknown profile: {profile}")


class DistanceGate:
    """
    radius_mm    : how far outside the mask the image survives (10 mm)
    pixel_mm     : in-plane pixel size of the resampled grid
    profile      : see falloff()
    z_offsets_mm : per-slice through-plane offsets, e.g. (-5, 0, 5). When given,
                   the distance is sqrt(d_inplane^2 + z^2), so the gate is a
                   sphere of radius R rather than a cylinder. With R = 10 mm and
                   a 5 mm gap the neighbouring slices then keep at most w = 0.5
                   and only an 8.66 mm in-plane radius, which is aggressive --
                   the default (None) applies the same in-plane gate to every
                   slice, treating the three planes as one 2D context.
    floor        : minimum weight far from the mask. 0 blanks the surroundings
                   completely; a small value (0.05-0.1) keeps a faint trace of
                   the anatomy and is worth trying if the target depends on
                   context.
    """

    def __init__(
        self,
        radius_mm: float = 10.0,
        pixel_mm: float = TARGET_INPLANE_MM,
        profile: str = "cosine",
        z_offsets_mm: Optional[Sequence[float]] = None,
        floor: float = 0.0,
    ):
        self.radius_mm = float(radius_mm)
        self.pixel_mm = float(pixel_mm)
        self.profile = profile
        self.z_offsets_mm = None if z_offsets_mm is None else tuple(z_offsets_mm)
        self.floor = float(floor)

    # -- distance ----------------------------------------------------------- #
    def distance_mm(self, mask: np.ndarray) -> np.ndarray:
        """In-plane distance in mm from every pixel to the nearest mask pixel."""
        m = np.asarray(mask) > 0
        if not m.any():
            return np.full(m.shape, np.inf, dtype=np.float32)
        # distance_transform_edt measures to the nearest ZERO, so invert.
        return distance_transform_edt(
            ~m, sampling=(self.pixel_mm, self.pixel_mm)).astype(np.float32)

    # -- weight map --------------------------------------------------------- #
    def weight(self, mask: np.ndarray, n_slices: int = 1) -> np.ndarray:
        """(n_slices, H, W) weights in [floor, 1]."""
        d = self.distance_mm(mask)
        if self.z_offsets_mm is None:
            d3 = np.repeat(d[None], n_slices, axis=0)
        else:
            if len(self.z_offsets_mm) != n_slices:
                raise ValueError("z_offsets_mm length must equal n_slices")
            z = np.asarray(self.z_offsets_mm, dtype=np.float32)[:, None, None]
            d3 = np.sqrt(d[None] ** 2 + z ** 2)
        t = np.clip(d3 / self.radius_mm, 0.0, 1.0)
        w = falloff(t, self.profile).astype(np.float32)
        return self.floor + (1.0 - self.floor) * w

    # -- apply -------------------------------------------------------------- #
    def __call__(self, image01: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """image01: (S, H, W) already normalized so that 0 = background."""
        return image01 * self.weight(mask, n_slices=image01.shape[0])


def window_01(x: np.ndarray, window: tuple[float, float] = CT_WINDOW) -> np.ndarray:
    """HU -> [0, 1]. Gating must happen in THIS domain, not in HU."""
    lo, hi = window
    return (np.clip(x, lo, hi) - lo) / (hi - lo)


def gate_image(
    image_hu: np.ndarray,
    mask: np.ndarray,
    gate: DistanceGate,
    window: tuple[float, float] = CT_WINDOW,
    out_range: tuple[float, float] = (-1.0, 1.0),
) -> np.ndarray:
    """
    HU -> window -> [0, 1] -> multiply by w -> rescale to out_range.

    Why not multiply the HU values directly: HU 0 is water, not "nothing", so
    0.5 * (-100 HU) = -50 HU is simply a different tissue, and the taper would
    write fat-like values into the fade zone. Multiplying in [0, 1] makes the
    faded region converge to the bottom of the window (-100 HU here), which
    reads as a uniform background.

    Equivalently, if you prefer to stay in [-1, 1]:
        x' = w * (x - bg) + bg      with bg = -1
    which is what the final rescale amounts to.
    """
    x01 = window_01(image_hu, window)
    x01 = gate(x01, mask)
    lo, hi = out_range
    return (x01 * (hi - lo) + lo).astype(np.float32)


def signed_distance_channel(
    mask: np.ndarray,
    pixel_mm: float = TARGET_INPLANE_MM,
    clip_mm: float = 10.0,
) -> np.ndarray:
    """
    Signed distance, negative inside the structure, clipped to +-clip_mm and
    scaled to [-1, 1]. A strictly more informative replacement for the binary
    mask channel: it encodes how far every pixel is from the boundary, which is
    exactly the geometry a size-related target depends on, and it is smooth
    under interpolation.
    """
    m = np.asarray(mask) > 0
    if not m.any():
        return np.ones(m.shape, dtype=np.float32)
    d_out = distance_transform_edt(~m, sampling=(pixel_mm, pixel_mm))
    d_in = distance_transform_edt(m, sampling=(pixel_mm, pixel_mm))
    sdf = d_out - d_in
    return np.clip(sdf / clip_mm, -1.0, 1.0).astype(np.float32)


def make_input(
    stack_hu: np.ndarray,
    gate: Optional[DistanceGate] = None,
    window: tuple[float, float] = CT_WINDOW,
    mask_channel: str = "sdf",        # 'binary' | 'sdf' | 'gate' | 'none'
    keep_context: bool = False,
) -> np.ndarray:
    """
    stack_hu : (S + 1, H, W) -- S image slices in HU plus the binary mask,
               i.e. the output of build_sample_sitk() after augmentation.

    Returns the network input. Channel layout:
        [gated slices] (+ [un-gated centre slice] if keep_context) + [mask ch]

    keep_context adds the un-gated centre slice as one extra channel. Gating is
    destructive -- once the surroundings are multiplied away the network cannot
    recover them -- so if the target might depend on the surrounding anatomy
    (cortical bone next to a lesion, adjacent organs), giving the network both
    views costs one channel and removes the risk.
    """
    stack_hu = np.asarray(stack_hu, dtype=np.float32)
    img_hu, msk = stack_hu[:-1], stack_hu[-1]
    S = img_hu.shape[0]

    if gate is not None:
        img = gate_image(img_hu, msk, gate, window)
    else:
        img = window_01(img_hu, window) * 2.0 - 1.0

    chans = [img]
    if keep_context:
        ctr = window_01(img_hu[S // 2], window) * 2.0 - 1.0
        chans.append(ctr[None])

    if mask_channel == "binary":
        chans.append((msk > 0).astype(np.float32)[None])
    elif mask_channel == "sdf":
        chans.append(signed_distance_channel(msk, gate.pixel_mm if gate else
                                             TARGET_INPLANE_MM)[None])
    elif mask_channel == "gate":
        chans.append(gate.weight(msk, 1) if gate is not None
                     else (msk > 0).astype(np.float32)[None])
    elif mask_channel != "none":
        raise ValueError(f"unknown mask_channel: {mask_channel}")

    return np.concatenate(chans, axis=0).astype(np.float32)
