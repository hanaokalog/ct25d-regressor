"""Body-size checkpoints: weights, geometry, standardizers and sigma scales."""

from __future__ import annotations

import torch

from ..transforms import TargetStandardizer
from .model import BodySizeNet

__all__ = ["save_bodysize", "load_bodysize", "KIND"]

KIND = "ct25d.bodysize"
VERSION = 1


def save_bodysize(path, model, standardizers: dict, config: dict,
                  sigma_scales: dict, metrics: dict | None = None) -> None:
    torch.save({
        "kind": KIND, "version": VERSION,
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "standardizers": {k: s.state_dict() for k, s in standardizers.items()},
        "sigma_scales": {k: float(v) for k, v in sigma_scales.items()},
        "config": config, "metrics": dict(metrics or {}),
    }, str(path))


def load_bodysize(path, map_location="cpu"):
    """Returns (model in eval mode, standardizers, config, sigma_scales, metrics)."""
    ck = torch.load(str(path), map_location=map_location, weights_only=True)
    if ck.get("kind") != KIND or ck.get("version") != VERSION:
        raise ValueError(f"{path} is not a {KIND} v{VERSION} checkpoint")
    cfg = ck["config"]
    model = BodySizeNet(**cfg["model"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    stds = {k: TargetStandardizer().load_state_dict(v)
            for k, v in ck["standardizers"].items()}
    return model, stds, cfg, ck["sigma_scales"], ck["metrics"]
