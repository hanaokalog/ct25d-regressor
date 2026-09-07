"""Crop large volumes down to the neighbourhood of a small label.

A whole-body CT is hundreds of megabytes; the structure of interest occupies a
few thousand voxels. Reading the full array once per epoch, or even once per
case, is almost entirely wasted work.

The label has to be read in full -- that is where the bounding box comes from --
but it is binary and compresses to a fraction of the image. The image is then
read through `ImageFileReader` with an extract region, so only the requested
block is materialized. For .nii.gz the file still has to be decompressed
sequentially, since a gzip stream has no random access, but the full array is
never allocated.

Geometry is preserved: the crop keeps its origin and direction cosines, so
physical coordinates in the cropped pair mean exactly what they meant in the
original, and the downstream pipeline produces identical output either way.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import SimpleITK as sitk

from .constants import SLICE_GAP_MM, TARGET_INPLANE_MM
from .geometry import (
    available_labels,
    binarize_label,
    label_hit,
    mask_extent_mm,
    required_patch_size,
)

__all__ = ["required_margin_mm", "label_bbox_physical", "crop_pair"]


def required_margin_mm(
    crop_size: int,
    in_plane_mm: float = TARGET_INPLANE_MM,
    scale_min: float = 0.9,
    rotate_deg: float = 5.0,
    translate: float = 0.08,
) -> float:
    """
    In-plane margin that keeps the downstream patch inside the crop.

    Cropping too tightly is silent: the patch simply picks up the fill value
    outside the region, and the model trains on a black wedge that only appears
    for augmented samples. The terms are the patch half-width, the extra reach
    of the widest zoom-out, the corner swing of the largest rotation, and the
    largest translation.
    """
    half = 0.5 * crop_size * in_plane_mm / float(scale_min)
    a = math.radians(rotate_deg)
    half *= abs(math.cos(a)) + abs(math.sin(a))
    return float(math.ceil(half + translate * crop_size * in_plane_mm))


def _reference(origin, spacing, direction) -> sitk.Image:
    """A 1-voxel image carrying only the geometry, for index/point conversion."""
    ref = sitk.Image(1, 1, 1, sitk.sitkUInt8)
    ref.SetOrigin(tuple(float(v) for v in origin))
    ref.SetSpacing(tuple(float(v) for v in spacing))
    ref.SetDirection(tuple(float(v) for v in direction))
    return ref


def label_bbox_physical(
    mask: sitk.Image,
    label_value: int = 1,
    margin_mm: float = 40.0,
    margin_mm_z: float = 10.0,
) -> list:
    """
    The eight physical corners of the label's bounding box, grown by a margin.

    The margin is physical, so the same call gives the same region whatever the
    voxel size of the file it came from.
    """
    arr = sitk.GetArrayViewFromImage(mask)            # (z, y, x)
    hit = label_hit(arr, label_value)
    if not hit.any():
        raise ValueError(
            f"no voxel with label {label_value} in the mask; "
            f"labels present: {available_labels(mask)}")
    zz = np.flatnonzero(hit.any(axis=(1, 2)))
    yy = np.flatnonzero(hit.any(axis=(0, 2)))
    xx = np.flatnonzero(hit.any(axis=(0, 1)))

    sx, sy, sz = mask.GetSpacing()
    mx, my, mz = margin_mm / sx, margin_mm / sy, margin_mm_z / sz
    lo = (float(xx[0]) - mx, float(yy[0]) - my, float(zz[0]) - mz)
    hi = (float(xx[-1]) + mx, float(yy[-1]) + my, float(zz[-1]) + mz)

    corners = []
    for i in (lo[0], hi[0]):
        for j in (lo[1], hi[1]):
            for k in (lo[2], hi[2]):
                corners.append(mask.TransformContinuousIndexToPhysicalPoint((i, j, k)))
    return corners


def _index_region(ref: sitk.Image, size, corners: Sequence):
    """Axis-aligned index region of `corners` in the grid of `ref`, clamped."""
    idx = np.array([ref.TransformPhysicalPointToContinuousIndex(tuple(c))
                    for c in corners])
    lo = np.floor(idx.min(axis=0)).astype(int)
    hi = np.ceil(idx.max(axis=0)).astype(int)
    size = np.asarray(size, dtype=int)
    lo = np.clip(lo, 0, size - 1)
    hi = np.clip(hi, 0, size - 1)
    return lo.tolist(), (hi - lo + 1).tolist()


def _read_region(path: str, start, extent) -> sitk.Image:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    reader.SetExtractIndex([int(v) for v in start])
    reader.SetExtractSize([int(v) for v in extent])
    return reader.Execute()


def _file_geometry(path: str):
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return (reader.GetSize(), _reference(reader.GetOrigin(), reader.GetSpacing(),
                                         reader.GetDirection()))


def crop_pair(
    image_path,
    mask_path,
    out_image,
    out_mask,
    label_value: int = 1,
    margin_mm: float = 40.0,
    margin_mm_z: float = 10.0,
    compress: bool = True,
    binarize: bool = False,
) -> dict:
    """
    Write a cropped (image, mask) pair and return a summary of what happened.

    The mask is cropped on its own grid rather than resampled onto the image
    grid: the label is the ground truth the target was derived from, and
    resampling it here would quietly change its area. The two crops cover the
    same physical box, which is all the downstream code needs.

    A multi-label file keeps all of its labels by default -- only the bounding
    box is taken from `label_value` -- so one crop can serve several targets.
    Pass binarize=True to write the selected label alone as 0/1.
    """
    mask = sitk.ReadImage(str(mask_path))             # binary, cheap to read
    corners = label_bbox_physical(mask, label_value, margin_mm, margin_mm_z)
    label_extent = mask_extent_mm(mask, label_value)

    img_size, img_ref = _file_geometry(str(image_path))
    start, extent = _index_region(img_ref, img_size, corners)
    image_crop = _read_region(str(image_path), start, extent)

    m_start, m_extent = _index_region(mask, mask.GetSize(), corners)
    mask_crop = sitk.RegionOfInterest(mask, m_extent, m_start)

    if binarize:
        mask_crop = binarize_label(mask_crop, label_value)
        kept = int(np.count_nonzero(sitk.GetArrayViewFromImage(mask_crop)))
    else:
        kept = int(np.count_nonzero(
            label_hit(sitk.GetArrayViewFromImage(mask_crop), label_value)))
    total = int(np.count_nonzero(
        label_hit(sitk.GetArrayViewFromImage(mask), label_value)))
    if kept != total:
        raise RuntimeError(f"crop lost label voxels ({kept} of {total}); "
                           "this is a bug, not a margin problem")

    sitk.WriteImage(image_crop, str(out_image), compress)
    sitk.WriteImage(mask_crop, str(out_mask), compress)

    return {
        "image_voxels_before": int(np.prod(img_size)),
        "image_voxels_after": int(np.prod(extent)),
        "reduction": float(np.prod(img_size)) / float(np.prod(extent)),
        "crop_size_vox": list(map(int, extent)),
        "crop_start_vox": list(map(int, start)),
        "label_voxels": total,
        "labels_present": available_labels(mask),
        "extent_mm": [round(v, 2) for v in label_extent],
        "extent_inplane_mm": round(max(label_extent[0], label_extent[1]), 2),
        "required_crop_size": required_patch_size(
            max(label_extent[0], label_extent[1])),
    }


def default_margins(n_slices: int = 3, gap_mm: float = SLICE_GAP_MM) -> float:
    """Through-plane margin that covers the neighbouring slices, plus one gap."""
    return float(gap_mm * ((n_slices - 1) / 2.0 + 1.0))
