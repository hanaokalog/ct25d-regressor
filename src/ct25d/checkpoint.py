"""Checkpoints that carry everything inference needs.

A weights file alone is not enough here. The predictions live in standardized
space and the input depends on the resampling grid, the HU window, the gate
radius and the channel layout. Storing the weights without those is how a model
silently produces wrong numbers six months later, so the configuration, the
target standardizer and the fitted sigma scale all travel with the weights.

Only plain containers and tensors are written, so the file loads with
`weights_only=True`.
"""

from __future__ import annotations

from typing import Any

import torch

from .models import resnet18_cbam25d, resnet34_cbam25d
from .transforms import TargetStandardizer

__all__ = ["build_model", "save_checkpoint", "load_checkpoint", "FORMAT_VERSION"]

FORMAT_VERSION = 2          # 2 adds the classification task and temperature
_READABLE = (1, 2)

_ARCHITECTURES = {
    "resnet18": resnet18_cbam25d,
    "resnet34": resnet34_cbam25d,
}


def build_model(arch: dict) -> torch.nn.Module:
    """Instantiate a model from the `arch` block of a config."""
    arch = dict(arch)
    name = arch.pop("name", "resnet18")
    if name not in _ARCHITECTURES:
        raise ValueError(f"unknown architecture {name!r}; "
                         f"choose from {sorted(_ARCHITECTURES)}")
    return _ARCHITECTURES[name](**arch)


def save_checkpoint(
    path,
    model: torch.nn.Module,
    standardizer: TargetStandardizer | None,
    config: dict,
    sigma_scale: float = 1.0,
    metrics: dict | None = None,
    temperature: float = 1.0,
) -> None:
    """A classifier has no standardizer (pass None) and is calibrated by
    `temperature` instead of `sigma_scale`."""
    torch.save(
        {
            "format_version": FORMAT_VERSION,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "standardizer": (None if standardizer is None
                             else standardizer.state_dict()),
            "config": config,
            "sigma_scale": float(sigma_scale),
            "temperature": float(temperature),
            "metrics": dict(metrics or {}),
        },
        str(path),
    )


def load_checkpoint(path, map_location: Any = "cpu"):
    """
    Returns (model in eval mode, standardizer, config, sigma_scale, metrics).

    For a classifier the standardizer is None and the calibration temperature
    is in config["temperature"].
    """
    ckpt = torch.load(str(path), map_location=map_location, weights_only=True)
    if ckpt.get("format_version") not in _READABLE:
        raise ValueError(f"checkpoint format {ckpt.get('format_version')} "
                         f"is not one of {_READABLE}")
    model = build_model(ckpt["config"]["arch"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    std = (None if ckpt["standardizer"] is None
           else TargetStandardizer().load_state_dict(ckpt["standardizer"]))
    config = dict(ckpt["config"])
    config.setdefault("task", "regression")
    config["temperature"] = float(ckpt.get("temperature", 1.0))
    return model, std, config, float(ckpt["sigma_scale"]), ckpt["metrics"]
