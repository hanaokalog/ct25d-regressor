#!/usr/bin/env python
"""Crop a cohort down to the neighbourhood of each label.

    python examples/crop.py cases.csv cropped.csv --out-dir /data/crops

Reads the same CSV as train.py -- one row per case, with a grayscale volume
path and a label volume path -- writes a cropped .nii.gz pair per row, and
writes a new CSV with every original column plus the paths of the crops. Point
train.py at the new CSV and nothing else changes.

The label is read in full, since that is where the bounding box comes from, but
it is binary and small once compressed. Only the corresponding block of the
image is materialized.

Choose the margin with --for-crop-size rather than by hand: it works out how
far the augmented patch can reach and sets the margin to cover it. Cropping too
tightly fails silently -- the patch picks up the fill value outside the crop and
the model sees a black wedge that appears only on augmented samples.

    python examples/crop.py cases.csv cropped.csv --out-dir /data/crops \\
        --image-col ct --mask-col seg --for-crop-size 96 --jobs 8
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ct25d.constants import SLICE_GAP_MM, TARGET_INPLANE_MM
from ct25d.crop import crop_pair, default_margins, required_margin_mm


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="input CSV, one case per row")
    p.add_argument("csv_out", type=Path, help="CSV to write, with the crop paths")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="directory for the cropped .nii.gz files")

    g = p.add_argument_group("columns")
    g.add_argument("--image-col", default="image")
    g.add_argument("--mask-col", default="mask")
    g.add_argument("--image-out-col", default="image_crop",
                   help="name of the new column holding the cropped image path")
    g.add_argument("--mask-out-col", default="mask_crop")
    g.add_argument("--id-col", default=None,
                   help="column to use for output filenames (default: row number)")

    g = p.add_argument_group("region")
    g.add_argument("--margin-mm", type=float, default=None,
                   help="in-plane margin around the label bounding box")
    g.add_argument("--margin-mm-z", type=float, default=None,
                   help="through-plane margin (default: covers the stacked slices)")
    g.add_argument("--for-crop-size", type=int, default=None,
                   help="derive --margin-mm from the patch size train.py will use")
    g.add_argument("--in-plane-mm", type=float, default=TARGET_INPLANE_MM)
    g.add_argument("--n-slices", type=int, default=3)
    g.add_argument("--gap-mm", type=float, default=SLICE_GAP_MM)
    g.add_argument("--scale-min", type=float, default=0.9,
                   help="smallest augmentation zoom train.py will use")
    g.add_argument("--rotate-deg", type=float, default=5.0)
    g.add_argument("--translate", type=float, default=0.08)
    g.add_argument("--label-value", type=int, default=1,
                   help="value of the target structure in a multi-label file")
    g.add_argument("--binarize-mask", action="store_true",
                   help="write only the selected label, as 0/1; the default "
                        "keeps every label so one crop can serve several targets")

    g = p.add_argument_group("run")
    g.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    g.add_argument("--skip-existing", action="store_true",
                   help="leave crops that are already on disk alone")
    g.add_argument("--no-compress", action="store_true")
    g.add_argument("--keep-failed", action="store_true",
                   help="keep rows that could not be cropped, with empty paths")
    return p.parse_args(argv)


def _one(job):
    """Runs in a worker process; takes and returns only plain data."""
    (pos, image_path, mask_path, out_image, out_mask, label_value,
     margin_mm, margin_mm_z, compress, skip_existing, binarize) = job
    try:
        if skip_existing and Path(out_image).exists() and Path(out_mask).exists():
            return pos, {"skipped_existing": True}, None
        stats = crop_pair(image_path, mask_path, out_image, out_mask,
                          label_value=label_value, margin_mm=margin_mm,
                          margin_mm_z=margin_mm_z, compress=compress,
                          binarize=binarize)
        return pos, stats, None
    except Exception as err:                           # noqa: BLE001
        return pos, None, f"{type(err).__name__}: {err}"


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

    margin = args.margin_mm
    if args.for_crop_size is not None:
        derived = required_margin_mm(args.for_crop_size, args.in_plane_mm,
                                     args.scale_min, args.rotate_deg,
                                     args.translate)
        if margin is not None and margin < derived:
            print(f"[warn] --margin-mm {margin} is below the {derived} mm needed "
                  f"for a {args.for_crop_size} px patch; using {derived}")
        margin = max(margin or 0.0, derived)
        print(f"margin {margin} mm in plane, derived from a "
              f"{args.for_crop_size} px patch at {args.in_plane_mm} mm")
    if margin is None:
        margin = 40.0
    margin_z = (args.margin_mm_z if args.margin_mm_z is not None
                else default_margins(args.n_slices, args.gap_mm))
    print(f"margins: {margin} mm in plane, {margin_z} mm through plane")

    df = pd.read_csv(args.csv)
    for col in (args.image_col, args.mask_col):
        if col not in df.columns:
            sys.exit(f"column {col!r} not in the CSV (columns: {list(df.columns)})")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.csv}: {len(df)} rows -> {args.out_dir}")
    print(f"target label value: {args.label_value}"
          + (" (written as 0/1)" if args.binarize_mask
             else " (all labels kept in the crop)"))

    def stem(pos, row):
        if args.id_col and args.id_col in df.columns:
            return str(row[args.id_col]).replace("/", "_")
        return f"{pos:05d}"

    jobs = []
    for pos, (_, row) in enumerate(df.iterrows()):
        s = stem(pos, row)
        jobs.append((pos, str(row[args.image_col]), str(row[args.mask_col]),
                     str(args.out_dir / f"{s}_img.nii.gz"),
                     str(args.out_dir / f"{s}_seg.nii.gz"),
                     args.label_value, margin, margin_z,
                     not args.no_compress, args.skip_existing,
                     args.binarize_mask))

    t0 = time.time()
    results = {}
    failures = []
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            for i, (pos, stats, err) in enumerate(pool.map(_one, jobs), 1):
                results[pos] = (stats, err)
                if err:
                    failures.append((pos, err))
                if i % 25 == 0:
                    print(f"  {i}/{len(jobs)}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            pos, stats, err = _one(job)
            results[pos] = (stats, err)
            if err:
                failures.append((pos, err))
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    out = df.copy()
    out[args.image_out_col] = [jobs[p][3] if results[p][1] is None else ""
                               for p in range(len(jobs))]
    out[args.mask_out_col] = [jobs[p][4] if results[p][1] is None else ""
                              for p in range(len(jobs))]
    out["crop_ok"] = [results[p][1] is None for p in range(len(jobs))]
    out["mask_extent_mm"] = [
        (results[p][0] or {}).get("extent_inplane_mm", float("nan"))
        for p in range(len(jobs))]
    out["required_crop_size"] = [
        (results[p][0] or {}).get("required_crop_size", -1)
        for p in range(len(jobs))]

    ok = [results[p][0] for p in range(len(jobs))
          if results[p][1] is None and results[p][0]]
    reductions = [s["reduction"] for s in ok if "reduction" in s]
    n_new = len(reductions)
    n_skipped = sum(1 for s in ok if s.get("skipped_existing"))

    if not args.keep_failed:
        out = out[out["crop_ok"]].drop(columns=["crop_ok"])
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.csv_out, index=False)

    print(f"\ncropped {n_new}, reused {n_skipped}, failed {len(failures)} "
          f"in {time.time() - t0:.0f}s")
    if reductions:
        r = np.array(reductions)
        print(f"volume reduction: median {np.median(r):.0f}x, "
              f"range {r.min():.0f}-{r.max():.0f}x")

    need = np.array([s_["required_crop_size"] for s_ in ok
                     if "required_crop_size" in s_])
    if need.size:
        ext = np.array([s_["extent_inplane_mm"] for s_ in ok
                        if "extent_inplane_mm" in s_])
        print(f"\nstructure size in plane: median {np.median(ext):.0f} mm, "
              f"max {ext.max():.0f} mm")
        print("crop_size needed for train.py "
              f"(at {args.in_plane_mm} mm/px, gate 10 mm, augmentation):")
        for q in (50, 90, 99, 100):
            v = int(np.percentile(need, q))
            covered = float((need <= v).mean()) * 100
            print(f"  {q:3d}th percentile: {v:4d} px "
                  f"({v * args.in_plane_mm:.0f} mm FOV, covers {covered:.0f}%)")
        print(f"  --> use --crop-size {int(need.max())} to fit every case; "
              f"a smaller value clips the largest structures silently")
    for pos, err in failures[:10]:
        print(f"  row {pos}: {err}")
    if len(failures) > 10:
        print(f"  ... and {len(failures) - 10} more")
    print(f"wrote {args.csv_out} "
          f"({args.image_out_col}, {args.mask_out_col} added)")
    if failures and not args.keep_failed:
        print(f"[note] {len(failures)} failed rows were dropped; "
              f"use --keep-failed to keep them with empty paths")


if __name__ == "__main__":
    main()
