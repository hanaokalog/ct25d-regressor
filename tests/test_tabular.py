"""Stack cache: reused only for the settings that wrote it."""

import numpy as np
import pytest

from ct25d.tabular import load_stack_cache, save_stack_cache

PREP = dict(image_col="ct", mask_col="seg", crop_size=96, n_slices=3,
            gap_mm=5.0, slab_mm=5.0, in_plane_mm=0.78125, label_value=1)


def test_cache_round_trip(tmp_path):
    stacks = np.random.default_rng(0).normal(size=(3, 4, 8, 8)).astype(np.float32)
    kept = np.array([0, 2, 3])
    path = tmp_path / "c.npz"
    save_stack_cache(path, stacks, kept, PREP)
    s, k = load_stack_cache(path, dict(PREP))
    assert np.array_equal(s, stacks) and np.array_equal(k, kept)


def test_cache_with_other_settings_is_refused(tmp_path):
    path = tmp_path / "c.npz"
    save_stack_cache(path, np.zeros((1, 4, 8, 8)), np.array([0]), PREP)
    with pytest.raises(ValueError, match="slab_mm"):
        load_stack_cache(path, {**PREP, "slab_mm": 0.0})


def test_cache_without_settings_is_refused(tmp_path):
    path = tmp_path / "old.npz"
    np.savez_compressed(path, stacks=np.zeros((1, 4, 8, 8)), kept=np.array([0]))
    with pytest.raises(ValueError, match="older version"):
        load_stack_cache(path, PREP)
