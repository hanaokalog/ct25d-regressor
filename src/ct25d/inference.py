"""Single-case inference with a train.py checkpoint, for use inside a pipeline.

    p = Predictor("age_marrow_L1.pt")
    out = p.predict(ct_image, label_image)
    # regression:     {"mean", "sigma", "lo95", "hi95"}
    # classification: {"pred", "conf", "probs"}

The preprocessing (grid, slab, gate, window, channels) comes from the
checkpoint, exactly as in examples/eval.py.
"""

from __future__ import annotations

import numpy as np
import torch

from .checkpoint import load_checkpoint
from .data import SliceStackDataset
from .gating import DistanceGate
from .geometry import build_sample_sitk

__all__ = ["Predictor"]


class Predictor:
    def __init__(self, path, device="cpu"):
        self.model, self.std, self.cfg, self.sigma_scale, _ = load_checkpoint(
            path, map_location=device)
        self.model.to(device)
        self.device = device
        cfg = self.cfg
        self.classify = cfg.get("task", "regression") == "classification"
        self.gate = None if cfg["gate"] is None else DistanceGate(
            radius_mm=cfg["gate"]["radius_mm"], pixel_mm=cfg["gate"]["pixel_mm"],
            profile=cfg["gate"]["profile"], floor=cfg["gate"]["floor"],
            z_offsets_mm=cfg["gate"]["z_offsets_mm"])

    @property
    def target(self) -> str:
        return self.cfg["target"]

    def stack(self, image, mask) -> np.ndarray:
        """The (n_slices + 1, H, W) HU stack the checkpoint expects."""
        c = self.cfg
        return build_sample_sitk(image, mask, label_value=c["label_value"],
                                 n_slices=c["n_slices"], gap_mm=c["gap_mm"],
                                 slab_mm=c.get("slab_mm", 0.0),
                                 in_plane_mm=c["in_plane_mm"],
                                 crop_size=c["crop_size"])

    @torch.no_grad()
    def predict(self, image, mask) -> dict:
        c = self.cfg
        ds = SliceStackDataset(self.stack(image, mask)[None], np.zeros(1),
                               standardizer=self.std, augment=None, gate=self.gate,
                               target_scale_power=0, window=tuple(c["window"]),
                               mask_channel=c["mask_channel"],
                               keep_context=c["keep_context"],
                               task=c.get("task", "regression"))
        x = ds[0][0][None].to(self.device)
        self.model.eval()
        if self.classify:
            p = torch.softmax(self.model(x).float() / c["temperature"], 1)[0]
            p = p.cpu().numpy()
            return {"pred": int(p.argmax()), "conf": float(p.max()), "probs": p}
        mu, log_var = self.model(x)
        mu_z = mu.float().cpu().numpy().ravel()
        sd_z = torch.exp(0.5 * log_var).float().cpu().numpy().ravel() * self.sigma_scale
        mean, sigma = self.std.inverse_transform(mu_z, sd_z)
        lo, hi = self.std.interval(mu_z, sd_z, 1.96)
        return {"mean": float(mean[0]), "sigma": float(sigma[0]),
                "lo95": float(lo[0]), "hi95": float(hi[0])}
