"""SimpleITK front end: volumes and masks to 3-slice stacks in HU.

All geometry is resolved in physical space, so oblique direction cosines,
differing origins, and a mask stored on its own grid are handled correctly.
"""

from collections.abc import Sequence
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .constants import AIR_HU, SLICE_GAP_MM, TARGET_INPLANE_MM

__all__ = ["find_center_slice", "mask_area_mm2", "mask_centroid_index",
           "build_sample_sitk", "build_samples_sitk"]


def find_center_slice(mask: sitk.Image, label_value: int = 1) -> int:
    """
    Index (along the image's k axis) of the slice carrying the label.

    If the label spans several slices, the intensity-weighted centroid is used
    and rounded; the caller is told, because for this task the mask is expected
    on exactly one slice.
    """
    arr = sitk.GetArrayViewFromImage(mask)          # (z, y, x)
    hit = np.asarray(arr == label_value)
    per_slice = hit.reshape(hit.shape[0], -1).sum(axis=1)
    nz = np.flatnonzero(per_slice)
    if nz.size == 0:
        raise ValueError(f"no voxel with label {label_value} in the mask")
    if nz.size > 1:
        w = per_slice[nz].astype(np.float64)
        c = int(round(float((nz * w).sum() / w.sum())))
        print(f"[warn] label spans {nz.size} slices ({nz.min()}-{nz.max()}); "
              f"using slice {c}")
        return c
    return int(nz[0])


def mask_area_mm2(mask: sitk.Image, label_value: int = 1) -> float:
    """
    In-plane area of the label, from the ORIGINAL mask and its own spacing.
    Derive regression targets with this, before any resampling.
    """
    arr = sitk.GetArrayViewFromImage(mask)
    sx, sy = mask.GetSpacing()[0], mask.GetSpacing()[1]
    return float(np.count_nonzero(arr == label_value) * sx * sy)


def mask_centroid_index(mask: sitk.Image, label_value: int = 1
                        ) -> tuple[float, float, float]:
    """Continuous index (x, y, z) of the label centroid, in the mask's grid."""
    arr = sitk.GetArrayViewFromImage(mask)
    zz, yy, xx = np.nonzero(arr == label_value)
    if xx.size == 0:
        raise ValueError(f"no voxel with label {label_value} in the mask")
    return float(xx.mean()), float(yy.mean()), float(zz.mean())


def _resample_plane(
    image: sitk.Image,
    center_point: Sequence[float],
    out_size_xy: tuple[int, int],
    reference_direction: Sequence[float],
    in_plane_mm: float,
    interpolator: int,
    default_value: float,
) -> np.ndarray:
    """
    Resample one axial plane of `image` centred on the physical point
    `center_point`, with `in_plane_mm` pixels and the given in-plane axes.
    """
    ow, oh = int(out_size_xy[0]), int(out_size_xy[1])
    D = np.asarray(reference_direction, dtype=np.float64).reshape(3, 3)
    spacing = np.array([in_plane_mm, in_plane_mm, 1.0], dtype=np.float64)
    idx_center = np.array([(ow - 1) / 2.0, (oh - 1) / 2.0, 0.0])
    # physical = origin + D @ (spacing * index)  ->  solve for origin
    origin = np.asarray(center_point, dtype=np.float64) - D @ (spacing * idx_center)

    ref = sitk.Image(ow, oh, 1, sitk.sitkFloat32)
    ref.SetSpacing(tuple(spacing))
    ref.SetOrigin(tuple(origin))
    ref.SetDirection(tuple(np.asarray(reference_direction, dtype=np.float64).ravel()))

    out = sitk.Resample(sitk.Cast(image, sitk.sitkFloat32), ref, sitk.Transform(),
                        interpolator, float(default_value), sitk.sitkFloat32)
    return sitk.GetArrayFromImage(out)[0]           # (H, W)


def build_sample_sitk(
    image: sitk.Image,
    mask: sitk.Image,
    label_value: int = 1,
    center_index: Optional[int] = None,
    n_slices: int = 3,
    gap_mm: float = SLICE_GAP_MM,
    in_plane_mm: float = TARGET_INPLANE_MM,
    crop_size: Optional[int] = None,
    center_on_mask: bool = True,
    default_value: float = AIR_HU,
    mask_threshold: float = 0.5,
) -> np.ndarray:
    """
    image      : 3D sitk.Image (CT volume)
    mask       : 3D sitk.Image, label on ONE slice; may live on its own grid
    crop_size  : side length in resampled pixels of a square patch around the
                 structure. None keeps the whole in-plane field of view.
    returns    : (n_slices + 1, H, W) float32, HU values untouched

    The centre slice is taken from the mask unless `center_index` is given.
    """
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("both image and mask must be 3D")

    cz_mask = find_center_slice(mask, label_value) if center_index is None else None
    direction = image.GetDirection()

    # ---- in-plane centre, expressed as a continuous index in the image ----- #
    if center_on_mask:
        cx_m, cy_m, cz_m = mask_centroid_index(mask, label_value)
        z_m = cz_mask if cz_mask is not None else cz_m
        p = mask.TransformContinuousIndexToPhysicalPoint((cx_m, cy_m, float(z_m)))
        ci = image.TransformPhysicalPointToContinuousIndex(p)
        cx, cy = ci[0], ci[1]
        cz = ci[2] if center_index is None else float(center_index)
    else:
        W, H, Z = image.GetSize()
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        if center_index is not None:
            cz = float(center_index)
        else:
            p = mask.TransformContinuousIndexToPhysicalPoint((0.0, 0.0, float(cz_mask)))
            cz = image.TransformPhysicalPointToContinuousIndex(p)[2]

    # ---- output size ------------------------------------------------------- #
    W, H, Z = image.GetSize()
    sx, sy, sz = image.GetSpacing()
    if crop_size is None:
        ow = int(round(W * sx / in_plane_mm))
        oh = int(round(H * sy / in_plane_mm))
    else:
        ow = oh = int(crop_size)

    # ---- three planes, z clamped inside the volume ------------------------- #
    offsets = (np.arange(n_slices) - (n_slices - 1) / 2.0) * gap_mm
    planes = []
    for off in offsets:
        z = float(np.clip(cz + off / sz, 0.0, Z - 1))
        p = image.TransformContinuousIndexToPhysicalPoint((cx, cy, z))
        planes.append(_resample_plane(image, p, (ow, oh), direction, in_plane_mm,
                                      sitk.sitkLinear, default_value))

    # ---- mask on the centre slice ------------------------------------------ #
    p_center = image.TransformContinuousIndexToPhysicalPoint((cx, cy, float(cz)))
    m = _resample_plane(mask, p_center, (ow, oh), direction, in_plane_mm,
                        sitk.sitkLinear, 0.0)
    # bilinear + threshold rather than nearest: nearest quantizes the boundary
    # and makes the effective area jump by whole pixels.
    m = (m >= mask_threshold * label_value).astype(np.float32)

    return np.concatenate([np.stack(planes), m[None]], axis=0).astype(np.float32)


def build_samples_sitk(
    images: Sequence[sitk.Image],
    masks: Sequence[sitk.Image],
    **kwargs,
) -> np.ndarray:
    """Batch version. Requires every case to end up with the same H, W:
    pass crop_size unless all volumes share a field of view."""
    if len(images) != len(masks):
        raise ValueError("images and masks must have the same length")
    out = [build_sample_sitk(im, mk, **kwargs) for im, mk in zip(images, masks)]
    shapes = {a.shape for a in out}
    if len(shapes) > 1:
        raise ValueError(
            f"inconsistent output shapes {shapes}; pass crop_size to fix the "
            "patch size, or batch with a collate function that pads")
    return np.stack(out)
