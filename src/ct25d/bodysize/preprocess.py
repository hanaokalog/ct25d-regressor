"""Body-size inputs from a trunk CT: projections of the trunk plus the L1/L3 planes.

Everything is resampled into one patient-aligned (LPS) frame anchored on L3, so
the same anatomy lands on the same rows whatever the scanner grid was:

* projections: 2 mm pixels, 512 rows from L3 + 384 mm (row 0, cranial) down
  to L3 - 640 mm, 256 columns (512 mm) centred on the trunk. For each of the
  lateral (along x) and frontal (along y) directions: the maximum intensity,
  the mean intensity and the fraction of the ray that was actually imaged
  (inside the scanned z range and the reconstruction field of view).
* axial planes: L1 and L3, 1 mm pixels, 512 x 512, centred like the
  projections, each with its own validity map; also the planes one slice above
  and below, for jitter during training.

HU are clipped to the window and mapped to [0, 1]; anything not imaged reads
as air (0) with validity 0, so the network can tell "no body here" from "not
scanned". The projections are also computed for a few smaller reconstruction
fields of view, the training augmentation for scanners that cut the body off
(the projection of a cut volume is not a cut of the projection, so the
variants are precomputed).

The body trunk itself (arms and table removed) is the caller's job; sarco has
extract_trunk.trunk_mask for that.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BodySizeGeometry", "level_z_mm", "prepare_case", "save_case", "load_case",
           "FOV_RADII_MM"]

#: reconstruction field-of-view radii precomputed for augmentation (None = as scanned)
FOV_RADII_MM = (None, 250.0, 225.0, 200.0, 175.0)

#: voxels below this in the raw volume lie outside the reconstruction circle
OUTSIDE_FOV_HU = -2000.0


class BodySizeGeometry:
    def __init__(self, proj_mm=2.0, rows=512, cols=256, l3_row=192,
                 axial_mm=1.0, axial_size=512, window=(-1000.0, 1000.0)):
        self.proj_mm, self.rows, self.cols, self.l3_row = proj_mm, rows, cols, l3_row
        self.axial_mm, self.axial_size = axial_mm, axial_size
        self.window = tuple(window)

    def as_dict(self):
        return dict(proj_mm=self.proj_mm, rows=self.rows, cols=self.cols,
                    l3_row=self.l3_row, axial_mm=self.axial_mm,
                    axial_size=self.axial_size, window=list(self.window))

    def row_of(self, z_mm, z_l3_mm):
        """Projection row of a physical z (LPS, cranial is +z)."""
        return self.l3_row - (z_mm - z_l3_mm) / self.proj_mm


def level_z_mm(label_image) -> float:
    """Physical z (LPS) of the slice with the largest labelled area."""
    import SimpleITK as sitk

    a = sitk.GetArrayViewFromImage(label_image)
    area = (a > 0).reshape(a.shape[0], -1).sum(1)
    if area.max() == 0:
        raise ValueError("empty label")
    k = int(np.argmax(area))
    size = label_image.GetSize()
    return float(label_image.TransformContinuousIndexToPhysicalPoint(
        ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0, float(k)))[2])


def _normalize(hu, window):
    lo, hi = window
    return np.clip((hu - lo) / (hi - lo), 0.0, 1.0)


def _resample(image, origin, spacing, size, default, linear=True):
    import SimpleITK as sitk

    f = sitk.ResampleImageFilter()
    f.SetOutputOrigin(tuple(float(v) for v in origin))
    f.SetOutputSpacing(tuple(float(v) for v in spacing))
    f.SetSize(tuple(int(v) for v in size))
    f.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    f.SetDefaultPixelValue(float(default))
    f.SetInterpolator(sitk.sitkLinear if linear else sitk.sitkNearestNeighbor)
    return sitk.GetArrayFromImage(f.Execute(image))      # (z, y, x)


def prepare_case(raw, trunk_mask, z_l1_mm, z_l3_mm,
                 geometry: BodySizeGeometry | None = None,
                 fov_radii=FOV_RADII_MM) -> dict:
    """
    raw        : sitk.Image in HU, as scanned
    trunk_mask : bool (z, y, x) on raw's grid, True inside the body trunk
    z_l1_mm, z_l3_mm : physical z (LPS) of the L1 and L3 planes

    Returns a dict of float16 arrays:
      proj   (len(fov_radii), 6, rows, cols): MIP_x, mean_x, valid_x,
             MIP_y, mean_y, valid_y; x-projections are (z, y), y-projections (z, x)
      axial  (2, 3, 2, S, S): level (L1, L3) x offset (-1, 0, +1 slice) x
             (value, valid)
      axial_dist (S, S): in-plane distance (mm) from the reconstruction centre,
             to apply a smaller field of view to the axial planes
    and the scalars l1_row, l3_row (projection rows) and fov_radius_mm.
    """
    import SimpleITK as sitk

    g = geometry or BodySizeGeometry()
    hu = sitk.GetArrayFromImage(raw).astype(np.float32)
    imaged = (hu > OUTSIDE_FOV_HU).astype(np.float32)
    body = np.where(trunk_mask, hu, -1000.0).astype(np.float32)

    def like(a):
        im = sitk.GetImageFromArray(a)
        im.CopyInformation(raw)
        return im
    body_im, imaged_im = like(body), like(imaged)

    # in-plane centre: the trunk's bounding box on the L3 plane (whole volume
    # as a fallback); reconstruction centre: the middle of the raw grid
    size = raw.GetSize()
    k3 = int(round(raw.TransformPhysicalPointToContinuousIndex(
        raw.TransformContinuousIndexToPhysicalPoint(
            ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0, 0.0))[:2] + (z_l3_mm,))[2]))
    k3 = min(max(k3, 0), size[2] - 1)
    m = trunk_mask[k3] if trunk_mask[k3].any() else trunk_mask.any(0)
    ys, xs = np.nonzero(m)
    c_idx = ((xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0, float(k3))
    cx, cy, _ = raw.TransformContinuousIndexToPhysicalPoint(c_idx)
    rx, ry, _ = raw.TransformContinuousIndexToPhysicalPoint(
        ((size[0] - 1) / 2.0, (size[1] - 1) / 2.0, float(k3)))
    sp = raw.GetSpacing()
    fov_radius = 0.5 * min(size[0] * sp[0], size[1] * sp[1])

    # projection volume: rows go caudally from L3 + l3_row * proj_mm
    p = g.proj_mm
    z_top = z_l3_mm + g.l3_row * p
    origin = (cx - (g.cols - 1) / 2.0 * p, cy - (g.cols - 1) / 2.0 * p,
              z_top - (g.rows - 1) * p)
    vol = _resample(body_im, origin, (p, p, p), (g.cols, g.cols, g.rows), -1000.0)
    val = _resample(imaged_im, origin, (p, p, p), (g.cols, g.cols, g.rows), 0.0)
    vol, val = vol[::-1], val[::-1]                      # row 0 is cranial
    vol = _normalize(vol, g.window)
    xs_mm = origin[0] + p * np.arange(g.cols)
    ys_mm = origin[1] + p * np.arange(g.cols)
    dist = np.hypot(xs_mm[None, :] - rx, ys_mm[:, None] - ry)      # (y, x)

    proj = []
    for r in fov_radii:
        if r is None:
            v, ok = vol, val
        else:
            inside = (dist <= r)[None]
            ok = val * inside
            v = np.where(inside, vol, 0.0)
        v = np.where(ok > 0.5, v, 0.0)
        proj.append(np.stack([v.max(2), v.mean(2), ok.mean(2),
                              v.max(1), v.mean(1), ok.mean(1)]))
    proj = np.stack(proj).astype(np.float16)

    # axial planes at L1 and L3 and one slice either side
    a, S = g.axial_mm, g.axial_size
    dz = sp[2]
    a_origin_xy = (cx - (S - 1) / 2.0 * a, cy - (S - 1) / 2.0 * a)
    axial = np.zeros((2, 3, 2, S, S), np.float32)
    for i, z0 in enumerate((z_l1_mm, z_l3_mm)):
        for j, off in enumerate((-1, 0, 1)):
            o = a_origin_xy + (z0 + off * dz,)
            v = _resample(body_im, o, (a, a, 1.0), (S, S, 1), -1000.0)[0]
            ok = _resample(imaged_im, o, (a, a, 1.0), (S, S, 1), 0.0)[0]
            axial[i, j, 0] = np.where(ok > 0.5, _normalize(v, g.window), 0.0)
            axial[i, j, 1] = ok
    ax_mm = a_origin_xy[0] + a * np.arange(S)
    ay_mm = a_origin_xy[1] + a * np.arange(S)
    axial_dist = np.hypot(ax_mm[None, :] - rx, ay_mm[:, None] - ry)

    return dict(proj=proj, axial=axial.astype(np.float16),
                axial_dist=axial_dist.astype(np.float16),
                l1_row=float(g.row_of(z_l1_mm, z_l3_mm)), l3_row=float(g.l3_row),
                fov_radius_mm=float(fov_radius))


def save_case(path, case: dict, geometry: BodySizeGeometry | None = None) -> None:
    """Write a prepared case (uncompressed: it is read every epoch)."""
    import json

    g = geometry or BodySizeGeometry()
    np.savez(path, geometry=np.array(json.dumps(g.as_dict())), **case)


def load_case(path) -> dict:
    """A prepared case written by save_case, without the geometry record."""
    z = np.load(path)
    return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files if k != "geometry"}
