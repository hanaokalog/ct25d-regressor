"""Tabular front end: a DataFrame of file paths becomes an array of stacks.

One row is one case: a grayscale volume, a label volume, and the variables that
go with them. Volumes are not assumed to share a voxel size or a matrix size,
which is why `crop_size` is mandatory here -- without it every case would come
out at a different resolution-dependent shape and could not be batched.
"""

from __future__ import annotations

import json

import numpy as np

__all__ = ["prepare_stacks", "split_indices", "load_stack_cache",
           "save_stack_cache"]


def prepare_stacks(
    df,
    image_col: str = "image",
    mask_col: str = "mask",
    crop_size: int = 96,
    verbose: bool = True,
    **build_kwargs,
):
    """
    Returns (stacks, kept, failures).

    stacks   : (n_kept, n_slices + 1, crop_size, crop_size) float32, in HU
    kept     : positional indices into `df` of the rows that succeeded
    failures : list of (index, reason) for the rows that did not

    A row that cannot be read or has an empty label is skipped rather than
    aborting the run, because one bad segmentation should not cost a whole
    training job -- but every skip is reported, since a silent skip is how a
    cohort quietly shrinks.
    """
    import SimpleITK as sitk

    from .geometry import build_sample_sitk

    if crop_size is None:
        raise ValueError(
            "crop_size is required: volumes with different voxel sizes produce "
            "different output shapes without a fixed patch size")
    for col in (image_col, mask_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in the CSV "
                           f"(columns: {list(df.columns)})")

    stacks, kept, failures = [], [], []
    for pos, (idx, row) in enumerate(df.iterrows()):
        try:
            img = sitk.ReadImage(str(row[image_col]))
            lab = sitk.ReadImage(str(row[mask_col]))
            stacks.append(build_sample_sitk(img, lab, crop_size=crop_size,
                                            **build_kwargs))
            kept.append(pos)
        except Exception as err:                       # noqa: BLE001
            failures.append((idx, f"{type(err).__name__}: {err}"))
        if verbose and (pos + 1) % 25 == 0:
            print(f"  prepared {pos + 1}/{len(df)}", flush=True)

    if not stacks:
        raise RuntimeError("no usable rows; first failures: " + str(failures[:3]))
    if verbose and failures:
        print(f"  skipped {len(failures)} rows:")
        for idx, why in failures[:10]:
            print(f"    {idx}: {why}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    return np.stack(stacks), np.asarray(kept), failures


def split_indices(
    df,
    val_frac: float = 0.2,
    group_col: str | None = None,
    split_col: str | None = None,
    seed: int = 0,
):
    """
    Returns (train_pos, val_pos) as positional indices.

    `split_col` takes precedence and is used verbatim (values 'train'/'val').
    Otherwise a random split is drawn; with `group_col` the split is made over
    groups, so two slices from the same patient can never land on opposite
    sides. Leakage through repeated patients is the standard way a medical
    imaging result turns out to be optimistic.
    """
    n = len(df)
    if split_col is not None:
        if split_col not in df.columns:
            raise KeyError(f"column {split_col!r} not in the CSV")
        v = df[split_col].astype(str).str.lower().to_numpy()
        train = np.flatnonzero(np.isin(v, ["train", "training", "0"]))
        val = np.flatnonzero(np.isin(v, ["val", "valid", "validation", "1"]))
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"{split_col!r} must contain both train and val rows")
        return train, val

    rng = np.random.default_rng(seed)
    if group_col is not None:
        if group_col not in df.columns:
            raise KeyError(f"column {group_col!r} not in the CSV")
        groups = df[group_col].to_numpy()
        uniq = np.unique(groups)
        rng.shuffle(uniq)
        n_val = max(1, int(round(len(uniq) * val_frac)))
        val_groups = set(uniq[:n_val].tolist())
        is_val = np.array([g in val_groups for g in groups])
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)

    perm = rng.permutation(n)
    n_val = max(1, int(round(n * val_frac)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def save_stack_cache(path, stacks, kept, prep: dict) -> None:
    """Write prepared stacks together with the settings that produced them."""
    np.savez_compressed(path, stacks=stacks, kept=kept,
                        prep=np.array(json.dumps(prep, sort_keys=True)))


def load_stack_cache(path, prep: dict):
    """
    Returns (stacks, kept) from a cache written with the same settings.

    A cache is only valid for the preprocessing that wrote it, and a stale one
    is silent: the arrays load fine and the model trains on, or is evaluated
    with, planes that no longer match its configuration. So the settings are
    stored alongside and a mismatch is an error rather than a reuse.
    """
    z = np.load(path)
    if "prep" not in z.files:
        raise ValueError(
            f"{path} has no record of its preprocessing settings (written by "
            f"an older version); delete it so the stacks are rebuilt")
    stored = json.loads(str(z["prep"]))
    want = json.loads(json.dumps(prep, sort_keys=True))
    diff = {k: (stored.get(k), want.get(k)) for k in set(stored) | set(want)
            if stored.get(k) != want.get(k)}
    if diff:
        raise ValueError(
            f"{path} was written with different preprocessing settings "
            f"(cached, requested): {diff}; delete it or point --cache elsewhere")
    return z["stacks"], z["kept"]
