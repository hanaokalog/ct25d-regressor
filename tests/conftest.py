"""Shared fixtures: synthetic volumes whose geometry is known exactly.

Each slice is filled with a constant equal to 100 * slice_index, so the slice a
resampled plane came from is recoverable from its intensity. That makes the
geometry assertions exact rather than approximate.
"""

import numpy as np
import pytest

sitk = pytest.importorskip("SimpleITK")

from ct25d.constants import AIR_HU


def make_case(
    size=(200, 200, 24),
    inplane=0.9766,
    z_spacing=2.5,
    center_index=None,
    radius_px=12.0,
    center_yx=None,
    direction_flip=False,
    origin=(-100.0, -120.0, 30.0),
):
    """Returns (image, mask, center_index) as sitk.Image."""
    W, H, Z = size
    cz = Z // 2 if center_index is None else center_index
    cy, cx = (H / 2, W / 2) if center_yx is None else center_yx
    yy, xx = np.mgrid[0:H, 0:W]

    vol = np.stack([np.full((H, W), 100.0 * z, np.float32) for z in range(Z)])
    msk = np.zeros((Z, H, W), np.uint8)
    msk[cz] = ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius_px ** 2)

    img = sitk.GetImageFromArray(vol)
    lab = sitk.GetImageFromArray(msk)
    for it in (img, lab):
        it.SetSpacing((inplane, inplane, z_spacing))
        it.SetOrigin(origin)
        if direction_flip:
            it.SetDirection((-1, 0, 0, 0, -1, 0, 0, 0, 1))
    return img, lab, cz


@pytest.fixture
def case():
    return make_case()


@pytest.fixture
def stack_hu():
    """(4, H, W) HU stack with a disc mask, without going through sitk."""
    H = W = 96
    yy, xx = np.mgrid[0:H, 0:W]
    mask = ((yy - 48) ** 2 + (xx - 48) ** 2 <= 8.0 ** 2).astype(np.float32)
    img = np.stack([np.full((H, W), v, np.float32) for v in (200.0, 400.0, 600.0)])
    img[:, mask == 0] = AIR_HU / 4
    return np.concatenate([img, mask[None]]).astype(np.float32)
