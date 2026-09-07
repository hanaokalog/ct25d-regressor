"""Dataset: HU stacks in, network input and standardized target out.

The order of operations matters and is fixed here:

    augment (HU domain, binary mask)
        -> recompute the distance gate from the AUGMENTED mask
        -> window to [0, 1], gate, rescale to [-1, 1]
        -> standardize the target

The gate is recomputed rather than transformed along with the image: an
interpolated weight map would no longer match the transformed mask exactly, and
a Euclidean distance transform on a patch of this size costs microseconds.
"""

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import CT_WINDOW
from .gating import DistanceGate, make_input

__all__ = ["SliceStackDataset"]


class SliceStackDataset(Dataset):
    """
    stacks       : (N, S+1, H, W) float32 in HU -- output of build_sample_sitk
    targets      : (N,) raw values in physical units (mm, mm^2, HU, ...)
    standardizer : fitted TargetStandardizer (fit on the training split only)
    augment      : RandomAffine2D or None
    gate         : DistanceGate or None (None disables gating)
    target_scale_power : 0 scale-invariant, 1 length-like, 2 area-like. Zoom
                   augmentation changes the apparent size of the structure, so
                   a size-related label has to follow it.

    Returns (x, z) where x is the network input and z the standardized target.
    """

    def __init__(
        self,
        stacks,
        targets,
        standardizer,
        augment=None,
        gate: Optional[DistanceGate] = None,
        target_scale_power: int = 0,
        window: tuple[float, float] = CT_WINDOW,
        mask_channel: str = "sdf",
        keep_context: bool = False,
    ):
        if len(stacks) != len(targets):
            raise ValueError("stacks and targets must have the same length")
        self.stacks = stacks
        self.targets = np.asarray(targets, dtype=np.float32).ravel()
        self.std = standardizer
        self.augment = augment
        self.gate = gate
        self.power = int(target_scale_power)
        self.window = window
        self.mask_channel = mask_channel
        self.keep_context = keep_context

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def n_channels(self) -> int:
        """Number of input channels this dataset produces."""
        s = np.asarray(self.stacks[0]).shape[0] - 1
        extra = 1 if self.keep_context else 0
        mask = 0 if self.mask_channel == "none" else 1
        return s + extra + mask

    def __getitem__(self, i):
        x = torch.as_tensor(np.asarray(self.stacks[i]), dtype=torch.float32).clone()
        y = float(self.targets[i])

        if self.augment is not None:
            x, scale = self.augment(x)          # HU domain; mask stays binary
            if self.power:
                y = y * (float(scale) ** self.power)

        arr = make_input(x.numpy(), gate=self.gate, window=self.window,
                         mask_channel=self.mask_channel,
                         keep_context=self.keep_context)

        z = self.std.transform(np.array([y], dtype=np.float32))
        return torch.from_numpy(arr), torch.as_tensor(z, dtype=torch.float32)
