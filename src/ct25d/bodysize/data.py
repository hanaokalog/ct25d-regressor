"""Dataset over prepared body-size cases, with the scan-coverage augmentations."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import FOV_RADII_MM

__all__ = ["BodySizeDataset", "select_inputs"]


def select_inputs(case: dict, fov_index=0, offsets=(1, 1), trim=None):
    """
    The network inputs of one prepared case (see preprocess.prepare_case).

    fov_index : which precomputed field of view to use (0 = as scanned)
    offsets   : axial plane per level, 0/1/2 = one slice below/at/above
    trim      : (top_row, bottom_row) kept of the projections; the rest reads
                as not scanned
    """
    proj = case["proj"][fov_index].astype(np.float32)
    if trim is not None:
        top, bottom = trim
        proj[:, :top] = 0.0
        proj[:, bottom + 1:] = 0.0
    planes = []
    r = FOV_RADII_MM[fov_index]
    inside = None if r is None else (case["axial_dist"] <= r)
    for level in (0, 1):
        v, ok = case["axial"][level, offsets[level]].astype(np.float32)
        if inside is not None:
            v, ok = v * inside, ok * inside
        planes += [v, ok]
    return proj, np.stack(planes)


class BodySizeDataset(Dataset):
    """
    cases   : list of prepared-case dicts
    targets : (n, n_targets) standardized targets
    augment : random field of view (only radii smaller than the scanned one),
              random craniocaudal trim keeping L1 to L3, and +-1 slice jitter
              of the axial planes
    """

    def __init__(self, cases, targets, augment=False, p_trim=0.8, p_fov=0.5,
                 jitter=True, margin_rows=5):
        self.cases, self.targets = cases, np.asarray(targets, np.float32)
        self.augment, self.p_trim, self.p_fov = augment, p_trim, p_fov
        self.jitter, self.margin = jitter, margin_rows
        self.epoch = 0

    def set_epoch(self, epoch: int):
        """Call once per epoch so the draws change between epochs."""
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.cases)

    def _augmentation(self, c, i):
        # a generator kept on the dataset would repeat the same draws in every
        # forked worker; seed from (torch's per-worker seed, epoch, index)
        rng = np.random.default_rng([torch.initial_seed() % 2 ** 32, self.epoch, i])
        fov = 0
        if rng.random() < self.p_fov:
            smaller = [k for k, r in enumerate(FOV_RADII_MM)
                       if r is not None and r < c["fov_radius_mm"]]
            if smaller:
                fov = int(rng.choice(smaller))
        trim = None
        if rng.random() < self.p_trim:
            scanned = np.flatnonzero(c["proj"][0, 2].astype(np.float32).max(1) > 0)
            if len(scanned):
                first, last = int(scanned[0]), int(scanned[-1])
                hi_top = int(np.floor(c["l1_row"])) - self.margin
                lo_bottom = int(np.ceil(c["l3_row"])) + self.margin
                top = int(rng.integers(first, max(first, hi_top) + 1))
                bottom = int(rng.integers(min(lo_bottom, last), last + 1))
                trim = (top, bottom)
        offsets = tuple(int(rng.integers(0, 3)) if self.jitter else 1 for _ in (0, 1))
        return fov, offsets, trim

    def __getitem__(self, i):
        c = self.cases[i]
        if self.augment:
            fov, offsets, trim = self._augmentation(c, i)
            proj, axial = select_inputs(c, fov, offsets, trim)
        else:
            proj, axial = select_inputs(c)
        return (torch.from_numpy(proj), torch.from_numpy(axial),
                torch.from_numpy(self.targets[i]))
