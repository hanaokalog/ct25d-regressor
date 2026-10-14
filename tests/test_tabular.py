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


def test_flip_col_reverses_the_slices_of_marked_rows(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    pd = pytest.importorskip("pandas")
    from conftest import make_case
    from ct25d.tabular import is_true, prepare_stacks

    # slice k holds 100 k; the label sits on slice 12 of 40, so the reversed
    # volume puts it on slice 27 and the planes there hold 100 * 12 again,
    # but the plane above / below swap
    img, lab, _ = make_case(size=(64, 64, 40), z_spacing=5.0, center_index=12)
    ip, mp = tmp_path / "i.nii.gz", tmp_path / "m.nii.gz"
    sitk.WriteImage(img, str(ip)); sitk.WriteImage(lab, str(mp))
    df = pd.DataFrame(dict(image=[str(ip)] * 3, mask=[str(mp)] * 3, flip=["1", 0, "true"]))
    s, kept, _ = prepare_stacks(df, crop_size=32, slab_mm=0, verbose=False, flip_col="flip")
    assert np.allclose(s[0, 1], s[1, 1]) and np.allclose(s[0, 3], s[1, 3])      # centre plane, mask
    assert np.allclose(s[0, 0], s[1, 2]) and np.allclose(s[0, 2], s[1, 0])      # neighbours swapped
    assert np.allclose(s[0], s[2])
    with pytest.raises(KeyError):
        prepare_stacks(df, crop_size=32, verbose=False, flip_col="nope")
    assert [is_true(v) for v in (1, "1", "TRUE", "yes", 0, "0", "", np.nan, None)] == \
        [True] * 4 + [False] * 5
