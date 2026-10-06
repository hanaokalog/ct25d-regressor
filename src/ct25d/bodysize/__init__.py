"""Height and weight from a trunk CT (projections plus the L1/L3 planes).

    from ct25d.bodysize import prepare_case, predict_bodysize, load_bodysize

prepare_case turns a CT, its body-trunk mask and the L1/L3 positions into the
network inputs; examples/train_bodysize.py trains on a table of such cases.
"""

from .checkpoint import load_bodysize, save_bodysize
from .data import BodySizeDataset, SingleLevelDataset, drop_shift, select_inputs
from .model import TARGETS, BodySizeNet
from .predict import predict_bodysize
from .preprocess import (
    FOV_RADII_MM,
    BodySizeGeometry,
    level_z_mm,
    load_case,
    prepare_case,
    save_case,
)

__all__ = ["BodySizeDataset", "BodySizeGeometry", "BodySizeNet", "FOV_RADII_MM",
           "SingleLevelDataset", "TARGETS", "drop_shift", "level_z_mm",
           "load_bodysize", "load_case", "predict_bodysize", "prepare_case",
           "save_bodysize", "save_case", "select_inputs"]
