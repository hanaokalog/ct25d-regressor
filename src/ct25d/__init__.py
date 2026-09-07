"""ct25d -- 2.5D CT regression with mask-guided attention and predictive uncertainty.

Pipeline
--------
    sitk.Image volume + mask
        -> geometry.build_sample_sitk       (4, H, W) in HU, 0.78125 mm, 5 mm gap
        -> transforms.RandomAffine2D        augmentation, still in HU
        -> gating.make_input                distance gate, window, channel layout
        -> models.ResNet25DCBAMRegressor    mu, log_var
        -> losses.WarmupHeteroscedasticLoss Huber -> Gaussian NLL
        -> transforms.TargetStandardizer    back to physical units
        -> calibration.fit_sigma_scale      calibrated uncertainty
"""

from .constants import (
    AIR_HU,
    CT_WINDOW,
    PIXEL_AREA_MM2,
    SLICE_GAP_MM,
    TARGET_INPLANE_MM,
)

__version__ = "0.1.0"

__all__ = ["TARGET_INPLANE_MM", "SLICE_GAP_MM", "PIXEL_AREA_MM2",
           "CT_WINDOW", "AIR_HU", "__version__"]


def __getattr__(name):
    """Lazy re-export, so importing ct25d does not require torch or SimpleITK."""
    _map = {
        "build_sample_sitk": "geometry", "build_samples_sitk": "geometry",
        "find_center_slice": "geometry", "mask_area_mm2": "geometry",
        "mask_centroid_index": "geometry",
        "DistanceGate": "gating", "make_input": "gating", "falloff": "gating",
        "signed_distance_channel": "gating", "window_01": "gating",
        "gate_image": "gating",
        "RandomAffine2D": "transforms", "TargetStandardizer": "transforms",
        "rescale_target": "transforms",
        "resnet18_cbam25d": "models", "resnet34_cbam25d": "models",
        "ResNet25DCBAMRegressor": "models", "CBAM2D": "models",
        "GaussianNLL": "losses", "WarmupHeteroscedasticLoss": "losses",
        "SliceStackDataset": "data",
        "fit_sigma_scale": "calibration", "uncertainty_report": "calibration",
    }
    if name in _map:
        import importlib
        return getattr(importlib.import_module(f".{_map[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
