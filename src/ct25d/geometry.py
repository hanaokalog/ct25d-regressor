"""SimpleITK front end: volumes and masks to 3-slice stacks in HU.

All geometry is resolved in physical space, so oblique direction cosines,
differing origins, and a mask stored on its own grid are handled correctly.
"""

import math
from collections.abc import Sequence
from typing import Optional

import numpy as np
import SimpleITK as sitk

from .constants import AIR_HU, SLAB_MM, SLICE_GAP_MM, TARGET_INPLANE_MM

__all__ = ["available_labels", "label_hit", "binarize_label", "find_center_slice",
           "mask_area_mm2", "mask_centroid_index", "mask_extent_mm",
           "required_patch_size", "slab_offsets_mm", "build_sample_sitk",
           "build_samples_sitk"]


def available_labels(mask: sitk.Image, limit: int = 20) -> list:
    """The non-zero values present in a label image, for error messages."""
    vals = np.unique(np.asarray(sitk.GetArrayViewFromImage(mask)))
    vals = [v for v in vals.tolist() if v != 0]
    return vals[:limit]


def label_hit(arr: np.ndarray, label_value: int) -> np.ndarray:
    """
    Boolean mask of one label in a possibly multi-label array.

    Compared with a half-unit tolerance rather than by equality, because label
    files are sometimes stored as float and 3.0000001 == 3 is False.
    """
    return np.abs(np.asarray(arr, dtype=np.float64) - float(label_value)) < 0.5


def binarize_label(mask: sitk.Image, label_value: int = 1) -> sitk.Image:
    """
    Extract one label as a 0/1 image, keeping the geometry.

    This has to happen BEFORE any interpolation. Interpolating a multi-label
    image mixes neighbouring structures -- halfway between label 2 and label 4
    the interpolated value is 3, which is a different structure entirely -- so
    resampling first and thresholding afterwards silently produces the wrong
    region whenever labels touch.
    """
    binary = sitk.BinaryThreshold(
        sitk.Cast(mask, sitk.sitkFloat32),
        lowerThreshold=float(label_value) - 0.5,
        upperThreshold=float(label_value) + 0.5,
        insideValue=1, outsideValue=0)
    return sitk.Cast(binary, sitk.sitkUInt8)


def _require_label(mask: sitk.Image, label_value: int) -> None:
    raise ValueError(
        f"no voxel with label {label_value} in the mask; "
        f"labels present: {available_labels(mask)}")


def find_center_slice(mask: sitk.Image, label_value: int = 1) -> int:
    """
    Index (along the image's k axis) of the slice carrying the label.

    If the label spans several slices, the slice with the largest labelled area
    is used (the first one on a tie); the caller is told, because for this task
    the mask is expected on exactly one slice. A centroid would be wrong here:
    labels on slices 72 and 74 round to 73, which carries no label at all.
    """
    arr = sitk.GetArrayViewFromImage(mask)          # (z, y, x)
    hit = label_hit(arr, label_value)
    per_slice = hit.reshape(hit.shape[0], -1).sum(axis=1)
    nz = np.flatnonzero(per_slice)
    if nz.size == 0:
        _require_label(mask, label_value)
    if nz.size > 1:
        c = int(np.argmax(per_slice))
        print(f"[warn] label spans {nz.size} slices ({nz.min()}-{nz.max()}); "
              f"using slice {c}, the one with the largest area")
        return c
    return int(nz[0])


def mask_area_mm2(mask: sitk.Image, label_value: int = 1) -> float:
    """
    In-plane area of the label, from the ORIGINAL mask and its own spacing.
    Derive regression targets with this, before any resampling.
    """
    arr = sitk.GetArrayViewFromImage(mask)
    sx, sy = mask.GetSpacing()[0], mask.GetSpacing()[1]
    return float(np.count_nonzero(label_hit(arr, label_value)) * sx * sy)


def mask_centroid_index(mask: sitk.Image, label_value: int = 1
                        ) -> tuple[float, float, float]:
    """Continuous index (x, y, z) of the label centroid, in the mask's grid."""
    arr = sitk.GetArrayViewFromImage(mask)
    zz, yy, xx = np.nonzero(label_hit(arr, label_value))
    if xx.size == 0:
        _require_label(mask, label_value)
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


def mask_extent_mm(mask: sitk.Image, label_value: int = 1) -> tuple:
    """
    Physical size of the label's bounding box, as (x, y, z) in millimetres.

    Measured along the image axes. What the patch has to hold is the in-plane
    extent, max(x, y).
    """
    arr = sitk.GetArrayViewFromImage(mask)
    hit = label_hit(arr, label_value)
    if not hit.any():
        _require_label(mask, label_value)
    sx, sy, sz = mask.GetSpacing()
    zz = np.flatnonzero(hit.any(axis=(1, 2)))
    yy = np.flatnonzero(hit.any(axis=(0, 2)))
    xx = np.flatnonzero(hit.any(axis=(0, 1)))
    return (float((xx[-1] - xx[0] + 1) * sx),
            float((yy[-1] - yy[0] + 1) * sy),
            float((zz[-1] - zz[0] + 1) * sz))


def required_patch_size(
    extent_mm: float,
    in_plane_mm: float = TARGET_INPLANE_MM,
    gate_radius_mm: float = 10.0,
    scale_max: float = 1.1,
    rotate_deg: float = 5.0,
    translate: float = 0.08,
    multiple: int = 16,
) -> int:
    """
    Smallest `crop_size` that still contains the structure after augmentation.

    The patch is a fixed number of pixels at a fixed millimetre spacing, so its
    physical field of view is fixed too -- 96 px at 0.78125 mm is 75 mm. A
    structure wider than that is clipped, and clipping is silent: the mask
    simply stops at the patch edge and the area the model sees is wrong.

    The terms are the structure's own extent, the largest zoom-in, the corner
    swing of the largest rotation, the gate band that should stay visible
    around it, and the largest translation. Rounded up to a multiple of
    `multiple` so the downsampling stages divide evenly.
    """
    half = 0.5 * float(extent_mm) * float(scale_max)
    a = math.radians(rotate_deg)
    half *= abs(math.cos(a)) + abs(math.sin(a))
    half += float(gate_radius_mm)
    px = 2.0 * half / float(in_plane_mm) / max(1e-6, 1.0 - 2.0 * translate)
    return int(math.ceil(px / multiple) * multiple)


def slab_offsets_mm(slab_mm: Optional[float], z_spacing_mm: float) -> np.ndarray:
    """
    Through-plane sample positions, relative to a plane, whose mean stands in
    for a slab `slab_mm` thick.

    Each native slice is taken to represent its own spacing of tissue, so a
    slab is covered by round(slab_mm / spacing) samples one native spacing
    apart: 5 at 1 mm, 2 at 2.5 mm, 8 at 0.625 mm. That is the same averaging a
    scanner does when it reconstructs a thick slice from thin ones. At 5 mm or
    coarser it is a single sample at offset 0 -- the native slice already is
    the slab, and averaging it again would blur 5 mm data beyond what the same
    patient would give at 1 mm.

    The slice thickness proper is not used, because NIfTI and MetaImage do not
    record it. For overlapping reconstructions (thickness larger than the
    spacing) this overestimates the sample count and smooths slightly more.

    `slab_mm` of None or 0 disables averaging: one sample at offset 0.
    """
    if not slab_mm or slab_mm <= 0:
        return np.zeros(1)
    sz = float(z_spacing_mm)
    n = max(1, int(round(float(slab_mm) / sz)))
    return (np.arange(n) - (n - 1) / 2.0) * sz


def build_sample_sitk(
    image: sitk.Image,
    mask: sitk.Image,
    label_value: int = 1,
    center_index: Optional[int] = None,
    n_slices: int = 3,
    gap_mm: float = SLICE_GAP_MM,
    slab_mm: Optional[float] = SLAB_MM,
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
    slab_mm    : each image plane is the mean over a slab this thick, centred
                 on the plane, so thin-slice volumes give the same planes as
                 the same anatomy at 5 mm (see slab_offsets_mm). None or 0
                 takes the single interpolated plane instead.
    returns    : (n_slices + 1, H, W) float32, HU values untouched

    The centre slice is taken from the mask unless `center_index` is given.
    The mask channel is not averaged: it is the label on the centre plane.
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

    # ---- three planes, each a slab mean, z clamped inside the volume ------- #
    offsets = (np.arange(n_slices) - (n_slices - 1) / 2.0) * gap_mm
    sub = slab_offsets_mm(slab_mm, sz)
    planes = []
    for off in offsets:
        acc = np.zeros((oh, ow), dtype=np.float64)
        for s in sub:
            z = float(np.clip(cz + (off + s) / sz, 0.0, Z - 1))
            p = image.TransformContinuousIndexToPhysicalPoint((cx, cy, z))
            acc += _resample_plane(image, p, (ow, oh), direction, in_plane_mm,
                                   sitk.sitkLinear, default_value)
        planes.append(acc / len(sub))

    # ---- mask on the centre slice ------------------------------------------ #
    p_center = image.TransformContinuousIndexToPhysicalPoint((cx, cy, float(cz)))
    # Binarize first: interpolating a multi-label image would blend the target
    # with whatever labels touch it. After binarization the plane is resampled
    # bilinearly and thresholded, rather than sampled with nearest neighbour,
    # because nearest quantizes the boundary and makes the effective area jump
    # by whole pixels.
    m = _resample_plane(binarize_label(mask, label_value), p_center, (ow, oh),
                        direction, in_plane_mm, sitk.sitkLinear, 0.0)
    m = (m >= mask_threshold).astype(np.float32)

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
