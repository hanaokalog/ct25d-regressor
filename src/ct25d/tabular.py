"""Tabular front end: a DataFrame of file paths becomes an array of stacks.

One row is one case: a grayscale volume, a label volume, and the variables that
go with them. Volumes are not assumed to share a voxel size or a matrix size,
which is why `crop_size` is mandatory here -- without it every case would come
out at a different resolution-dependent shape and could not be batched.
"""

from __future__ import annotations

import json

import numpy as np

__all__ = ["read_table", "parse_path_map", "apply_path_map", "prepare_stacks",
           "split_indices", "split_three", "load_stack_cache", "save_stack_cache"]

ENCODINGS = ("utf-8-sig", "cp932")


def read_table(path, encoding: str | None = None, **kwargs):
    """
    Read a CSV whose encoding is not known in advance.

    Lists written on Windows arrive as UTF-8, UTF-8 with a BOM, or CP932
    (Shift_JIS), and a wrong guess either fails or -- worse -- reads a BOM into
    the first column name, so `id` silently becomes `\ufeffid`. `utf-8-sig`
    reads plain UTF-8 and strips a BOM, so trying it before CP932 covers all
    three. Column names and string cells are stripped of surrounding
    whitespace, which is how Excel exports tend to pad them.
    """
    import pandas as pd

    tried = []
    for enc in ([encoding] if encoding else ENCODINGS):
        try:
            df = pd.read_csv(path, encoding=enc, **kwargs)
            break
        except UnicodeDecodeError as err:
            tried.append(f"{enc}: {err}")
    else:
        raise ValueError(f"cannot decode {path}; tried " + "; ".join(tried))
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return df


def parse_path_map(specs) -> list[tuple[str, str]]:
    """'/home/hanaoka=/mnt/w' -> [('/home/hanaoka', '/mnt/w')]."""
    out = []
    for spec in specs or ():
        if "=" not in spec:
            raise ValueError(f"--path-map takes OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        out.append((old, new))
    return out


def apply_path_map(df, columns, mapping):
    """
    Rewrite path prefixes in `columns`, so a list written on one machine can be
    read on another where the same files are mounted elsewhere. Only a leading
    prefix that ends at a path separator is replaced. Returns a copy.
    """
    if not mapping:
        return df
    df = df.copy()

    def remap(v):
        if not isinstance(v, str):
            return v
        for old, new in mapping:
            o = old.rstrip("/")
            if v == o or v.startswith(o + "/"):
                return new.rstrip("/") + v[len(o):]
        return v

    for c in columns:
        if c in df.columns:
            df[c] = df[c].map(remap)
    return df


def prepare_stacks(
    df,
    image_col: str = "image",
    mask_col: str = "mask",
    crop_size: int = 96,
    verbose: bool = True,
    workers: int = 1,
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

    `workers` > 1 reads and resamples the rows in that many processes; reading
    dominates on network storage. The result is the same as with one.
    """
    if crop_size is None:
        raise ValueError(
            "crop_size is required: volumes with different voxel sizes produce "
            "different output shapes without a fixed patch size")
    for col in (image_col, mask_col):
        if col not in df.columns:
            raise KeyError(f"column {col!r} not in the CSV "
                           f"(columns: {list(df.columns)})")

    jobs = [(str(row[image_col]), str(row[mask_col]), crop_size, build_kwargs)
            for _, row in df.iterrows()]
    if workers > 1:
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        # spawn, not fork: the caller usually has torch threads running
        pool = ProcessPoolExecutor(workers,
                                   mp_context=multiprocessing.get_context("spawn"))
        results = pool.map(_prepare_one, jobs, chunksize=4)
    else:
        pool = None
        results = map(_prepare_one, jobs)

    stacks, kept, failures = [], [], []
    try:
        for pos, (idx, (stack, err)) in enumerate(zip(df.index, results)):
            if err is None:
                stacks.append(stack)
                kept.append(pos)
            else:
                failures.append((idx, err))
            if verbose and (pos + 1) % 25 == 0:
                print(f"  prepared {pos + 1}/{len(df)}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    if not stacks:
        raise RuntimeError("no usable rows; first failures: " + str(failures[:3]))
    if verbose and failures:
        print(f"  skipped {len(failures)} rows:")
        for idx, why in failures[:10]:
            print(f"    {idx}: {why}")
        if len(failures) > 10:
            print(f"    ... and {len(failures) - 10} more")
    return np.stack(stacks), np.asarray(kept), failures


def _prepare_one(job):
    """(stack, None) or (None, reason) for one row; top level so it pickles."""
    import SimpleITK as sitk

    from .geometry import build_sample_sitk

    image_path, mask_path, crop_size, build_kwargs = job
    try:
        img = sitk.ReadImage(image_path)
        lab = sitk.ReadImage(mask_path)
        return build_sample_sitk(img, lab, crop_size=crop_size, **build_kwargs), None
    except Exception as err:                           # noqa: BLE001
        return None, f"{type(err).__name__}: {err}"


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


def split_three(
    df,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    group_col: str | None = None,
    split_col: str | None = None,
    seed: int = 0,
):
    """
    Returns (train_pos, val_pos, test_pos) as positional indices.

    As split_indices, with a held-out test split that neither training nor
    calibration ever sees. `split_col` is used verbatim when given, with values
    'train' / 'val' / 'test'; rows with any other value are left out of all
    three. Otherwise the groups (or rows) are shuffled once with `seed` and cut
    into test, val and train in that order, so the same seed always gives the
    same split.
    """
    if split_col is not None:
        if split_col not in df.columns:
            raise KeyError(f"column {split_col!r} not in the CSV")
        v = df[split_col].astype(str).str.strip().str.lower().to_numpy()
        train = np.flatnonzero(np.isin(v, ["train", "training"]))
        val = np.flatnonzero(np.isin(v, ["val", "valid", "validation"]))
        test = np.flatnonzero(np.isin(v, ["test"]))
        if len(train) == 0 or len(val) == 0:
            raise ValueError(f"{split_col!r} must contain both train and val rows")
        return train, val, test

    rng = np.random.default_rng(seed)
    if group_col is not None:
        if group_col not in df.columns:
            raise KeyError(f"column {group_col!r} not in the CSV")
        groups = df[group_col].astype(str).to_numpy()
    else:
        groups = np.arange(len(df)).astype(str)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_frac))
    n_val = max(1, int(round(len(uniq) * val_frac)))
    test_g = set(uniq[:n_test].tolist())
    val_g = set(uniq[n_test:n_test + n_val].tolist())
    which = np.array([2 if g in test_g else 1 if g in val_g else 0 for g in groups])
    return (np.flatnonzero(which == 0), np.flatnonzero(which == 1),
            np.flatnonzero(which == 2))


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
