"""Dataset over prepared body-size cases, with the scan-coverage augmentations."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import FOV_RADII_MM

__all__ = ["BodySizeDataset", "SingleLevelDataset", "select_inputs", "drop_shift"]


def drop_shift(case: dict, l1_l3_rows: float) -> int:
    """
    Rows by which the projections move when L3 is dropped: the frame is then
    anchored on L3 as estimated from L1 (l1_row + l1_l3_rows), as
    prepare_case does for a scan without L3.
    """
    return int(round(case["l1_row"] + l1_l3_rows - case["l3_row"]))


def select_inputs(case: dict, fov_index=0, offsets=(1, 1), trim=None, drop=None,
                  l1_l3_rows=None):
    """
    The network inputs of one prepared case (see preprocess.prepare_case).

    fov_index : which precomputed field of view to use (0 = as scanned)
    offsets   : axial plane per level, 0/1/2 = one slice below/at/above
    trim      : (top_row, bottom_row) kept of the projections; the rest reads
                as not scanned (rows of the frame after any drop shift)
    drop      : None, "L1" or "L3" -- treat that level as not found (training
                augmentation for scans that miss it): its axial planes are
                emptied, and for "L3" the projections are re-anchored on the L3
                estimated from L1 (needs l1_l3_rows)
    A level the case itself lacks (has_l1 / has_l3 = 0) is always empty.
    """
    proj = case["proj"][fov_index].astype(np.float32)
    if drop == "L3":
        shift = drop_shift(case, l1_l3_rows)
        moved = np.zeros_like(proj)
        n = proj.shape[1]
        src = np.arange(n) + shift
        ok = (src >= 0) & (src < n)
        moved[:, ok] = proj[:, src[ok]]
        proj = moved
    if trim is not None:
        top, bottom = trim
        proj[:, :max(top, 0)] = 0.0
        proj[:, bottom + 1:] = 0.0
    present = [bool(case.get("has_l1", 1.0)) and drop != "L1",
               bool(case.get("has_l3", 1.0)) and drop != "L3"]
    planes = []
    r = FOV_RADII_MM[fov_index]
    inside = None if r is None else (case["axial_dist"] <= r)
    for level in (0, 1):
        v, ok = case["axial"][level, offsets[level]].astype(np.float32)
        if not present[level]:
            v, ok = np.zeros_like(v), np.zeros_like(ok)
        elif inside is not None:
            v, ok = v * inside, ok * inside
        planes += [v, ok]
    return proj, np.stack(planes)


class BodySizeDataset(Dataset):
    """
    cases   : list of prepared-case dicts
    targets : (n, n_targets) standardized targets
    augment : random field of view (only radii smaller than the scanned one),
              random craniocaudal trim keeping L1 to L3, +-1 slice jitter of
              the axial planes, and with p_drop each a scan that misses L1
              (it starts between L1 and L3) or L3 (it ends between them)
    l1_l3_rows : geometry.l1_l3_mm / proj_mm, for the L3 estimated from L1
    """

    def __init__(self, cases, targets, augment=False, p_trim=0.8, p_fov=0.5,
                 jitter=True, margin_rows=5, p_drop=0.0, l1_l3_rows=32.5):
        self.cases, self.targets = cases, np.asarray(targets, np.float32)
        self.augment, self.p_trim, self.p_fov = augment, p_trim, p_fov
        self.jitter, self.margin = jitter, margin_rows
        self.p_drop, self.l1_l3_rows = p_drop, l1_l3_rows
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
        u = rng.random()
        drop = "L1" if u < self.p_drop else ("L3" if u < 2 * self.p_drop else None)
        shift = drop_shift(c, self.l1_l3_rows) if drop == "L3" else 0
        l1, l3 = c["l1_row"] - shift, c["l3_row"] - shift     # rows of the frame used
        scanned = np.flatnonzero(c["proj"][0, 2].astype(np.float32).max(1) > 0) - shift
        trim = None
        if len(scanned) and (drop or rng.random() < self.p_trim):
            first, last = int(scanned[0]), int(scanned[-1])
            m = self.margin
            if drop == "L3":       # the scan ends between L1 and L3
                top_hi, bot_lo, bot_hi = int(l1) - m, int(l1) + m, int(l3) - 1
            elif drop == "L1":     # the scan starts between L1 and L3
                first = max(first, int(np.ceil(l1)) + 1)
                top_hi, bot_lo, bot_hi = int(l3) - m, int(l3) + m, last
            else:                  # L1 to L3 kept
                top_hi = int(np.floor(l1)) - m
                bot_lo, bot_hi = int(np.ceil(l3)) + m, last
            top = int(rng.integers(first, max(first, top_hi) + 1))
            bot_lo = min(bot_lo, last)
            bottom = int(rng.integers(bot_lo, max(bot_lo, min(bot_hi, last)) + 1))
            trim = (top, bottom)
        offsets = tuple(int(rng.integers(0, 3)) if self.jitter else 1 for _ in (0, 1))
        return fov, offsets, trim, drop

    def __getitem__(self, i):
        c = self.cases[i]
        if self.augment:
            fov, offsets, trim, drop = self._augmentation(c, i)
            proj, axial = select_inputs(c, fov, offsets, trim, drop, self.l1_l3_rows)
        else:
            proj, axial = select_inputs(c)
        return (torch.from_numpy(proj), torch.from_numpy(axial),
                torch.from_numpy(self.targets[i]))


class SingleLevelDataset(Dataset):
    """
    Evaluation: every case as a scan that shows only `kept` ("L1" or "L3"),
    cut halfway between the two levels, without other augmentation.
    """

    def __init__(self, cases, targets, kept, l1_l3_rows=32.5):
        self.cases, self.targets = cases, np.asarray(targets, np.float32)
        self.kept, self.l1_l3_rows = kept, l1_l3_rows

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, i):
        c = self.cases[i]
        drop = "L3" if self.kept == "L1" else "L1"
        shift = drop_shift(c, self.l1_l3_rows) if drop == "L3" else 0
        mid = int(round((c["l1_row"] + c["l3_row"]) / 2)) - shift
        n = c["proj"].shape[2]
        trim = (0, mid) if drop == "L3" else (mid + 1, n - 1)
        proj, axial = select_inputs(c, trim=trim, drop=drop, l1_l3_rows=self.l1_l3_rows)
        return (torch.from_numpy(proj), torch.from_numpy(axial),
                torch.from_numpy(self.targets[i]))
