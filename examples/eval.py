#!/usr/bin/env python
"""Predict and evaluate with a checkpoint from train.py.

    python examples/eval.py cases.csv lesion_area_mm2 model.pt

Same three positional arguments as train.py, with the model read rather than
written. Every preprocessing setting -- grid, window, gate, channel layout --
comes from the checkpoint, not from flags, so evaluation cannot silently drift
from training. The only thing taken from the command line is where the data is.

If the target column is absent from the CSV the script still runs and writes
predictions; it just skips the accuracy and calibration report. To predict on
unlabelled cases, pass the column name anyway:

    python examples/eval.py new_cases.csv lesion_area_mm2 model.pt \\
        --out predictions.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct25d.calibration import fit_sigma_scale, uncertainty_report
from ct25d.checkpoint import load_checkpoint
from ct25d.data import SliceStackDataset
from ct25d.gating import DistanceGate
from ct25d.tabular import load_stack_cache, prepare_stacks, save_stack_cache


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="CSV with one case per row")
    p.add_argument("target", help="column holding the ground truth, if present")
    p.add_argument("model", type=Path, help="checkpoint written by train.py")
    p.add_argument("--out", type=Path, default=None,
                   help="write per-case predictions here (default: <csv>.pred.csv)")
    p.add_argument("--image-col", default=None, help="override the trained column")
    p.add_argument("--mask-col", default=None, help="override the trained column")
    p.add_argument("--cache", type=Path, default=None)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-calibration", action="store_true",
                   help="report the raw sigma instead of the calibrated one")
    p.add_argument("--refit-calibration", action="store_true",
                   help="refit the sigma scale on THIS set; only valid if it is a "
                        "held-out calibration split, never the test set you report")
    return p.parse_args(argv)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    mus, sigmas = [], []
    for batch in loader:
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        mu, log_var = model(x.to(device))
        mus.append(mu.cpu().view(-1))
        sigmas.append(torch.exp(0.5 * log_var).cpu().view(-1))
    return torch.cat(mus).numpy(), torch.cat(sigmas).numpy()


def main(argv=None):
    args = parse_args(argv)
    import pandas as pd

    model, std, cfg, sigma_scale, train_metrics = load_checkpoint(
        args.model, map_location=args.device)
    model.to(args.device)
    print(f"{args.model}: trained on {cfg['target']!r}, "
          f"{cfg['crop_size']}px crop, {cfg['n_slices']} slices, "
          f"sigma scale {sigma_scale:.3f}")
    if train_metrics:
        print("  training run reported "
              + ", ".join(f"{k}={v:.4g}" for k, v in train_metrics.items()
                          if isinstance(v, (int, float))))
    if cfg["target"] != args.target:
        print(f"[warn] the checkpoint was trained on {cfg['target']!r}, "
              f"evaluating against {args.target!r}")

    df = pd.read_csv(args.csv)
    has_truth = args.target in df.columns
    if has_truth:
        df = df[df[args.target].notna()].reset_index(drop=True)
    else:
        print(f"[info] {args.target!r} not in the CSV; predicting only")
    print(f"{args.csv}: {len(df)} rows")

    image_col = args.image_col or cfg["image_col"]
    mask_col = args.mask_col or cfg["mask_col"]
    prep = dict(image_col=image_col, mask_col=mask_col,
                crop_size=cfg["crop_size"], n_slices=cfg["n_slices"],
                gap_mm=cfg["gap_mm"], in_plane_mm=cfg["in_plane_mm"],
                # checkpoints from before slab averaging took a single plane
                slab_mm=cfg.get("slab_mm", 0.0),
                label_value=cfg["label_value"])
    if args.cache is not None and args.cache.exists():
        try:
            stacks, kept = load_stack_cache(args.cache, prep)
        except ValueError as err:
            raise SystemExit(f"[error] {err}") from None
        print(f"loaded cached stacks from {args.cache}")
    else:
        stacks, kept, _ = prepare_stacks(df, **prep)
        if args.cache is not None:
            args.cache.parent.mkdir(parents=True, exist_ok=True)
            save_stack_cache(args.cache, stacks, kept, prep)
    df = df.iloc[kept].reset_index(drop=True)

    gate = None if cfg["gate"] is None else DistanceGate(
        radius_mm=cfg["gate"]["radius_mm"], pixel_mm=cfg["gate"]["pixel_mm"],
        profile=cfg["gate"]["profile"], floor=cfg["gate"]["floor"],
        z_offsets_mm=cfg["gate"]["z_offsets_mm"])

    # Targets only feed the report; the dataset needs an array either way.
    y = (df[args.target].to_numpy(dtype=np.float64) if has_truth
         else np.zeros(len(df)))
    ds = SliceStackDataset(stacks, y, standardizer=std, augment=None, gate=gate,
                           target_scale_power=0, window=tuple(cfg["window"]),
                           mask_channel=cfg["mask_channel"],
                           keep_context=cfg["keep_context"])
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.num_workers)

    mu_z, sigma_z = predict(model, loader, args.device)

    scale = 1.0 if args.no_calibration else sigma_scale
    if args.refit_calibration:
        if not has_truth:
            raise SystemExit("--refit-calibration needs the target column")
        scale = fit_sigma_scale(mu_z, sigma_z, std.transform(y))
        print(f"refit sigma scale on this set: {scale:.3f}")
    mean, sigma = std.inverse_transform(mu_z, sigma_z * scale)

    out = df.copy()
    out["pred"] = mean
    out["pred_sigma"] = sigma
    out["pred_lo95"] = mean - 1.96 * sigma
    out["pred_hi95"] = mean + 1.96 * sigma
    if has_truth:
        out["residual"] = y - mean
        out["z_score"] = (y - mean) / sigma

    path = args.out or args.csv.with_suffix(".pred.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path}")

    if not has_truth:
        print(f"\npredicted {cfg['target']}: {mean.min():.4g} .. {mean.max():.4g}, "
              f"median sigma {np.median(sigma):.4g}")
        return

    rep = uncertainty_report(mean, sigma, y)
    print(f"\nevaluation, in the units of {args.target}:")
    for k, v in rep.items():
        print(f"  {k:14s} {v:.4f}")
    if abs(rep["z_std"] - 1.0) > 0.2:
        direction = "over" if rep["z_std"] > 1.0 else "under"
        print(f"\n[warn] z_std {rep['z_std']:.2f}: the model is {direction}confident "
              f"on this set. Sigma fitted on the training run's validation split "
              f"does not always transfer; check for a distribution shift before "
              f"trusting the intervals.")
    if rep["corr_abs_err"] < 0.2:
        print("[warn] sigma correlates weakly with the actual error, so it is "
              "close to a constant and adds little per case.")


if __name__ == "__main__":
    main()
