#!/usr/bin/env python
"""Train from a CSV of volume/label paths.

    python examples/train.py cases.csv lesion_area_mm2 model.pt

The CSV has one row per case: a path to a grayscale 3D image, a path to a label
image whose structure sits on a single slice, and any number of variables. The
second argument names the column to regress. The third is where the checkpoint
is written -- weights, preprocessing configuration, target standardizer and
calibrated sigma scale in one file, so eval.py needs nothing else.

Volumes need not share a voxel size or a matrix size; everything is resampled
to a common physical grid, and --crop-size fixes the patch shape.

    python examples/train.py cases.csv volume_mm3 model.pt \\
        --image-col ct --mask-col seg --group-col patient_id \\
        --target-power 2 --log-target --epochs 120

With --task classification the target column holds class indices 0..K-1, the
head outputs K logits trained with cross-entropy, and the confidence is
calibrated with temperature scaling on the validation split.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct25d.calibration import (
    classification_report,
    fit_sigma_scale,
    fit_temperature,
    uncertainty_report,
)
from ct25d.checkpoint import build_model, save_checkpoint
from ct25d.constants import CT_WINDOW, SLAB_MM, SLICE_GAP_MM, TARGET_INPLANE_MM
from ct25d.data import SliceStackDataset
from ct25d.gating import DistanceGate
from ct25d.geometry import required_patch_size
from ct25d.losses import WarmupHeteroscedasticLoss
from ct25d.tabular import (
    apply_path_map,
    load_stack_cache,
    parse_path_map,
    prepare_stacks,
    read_table,
    save_stack_cache,
    split_three,
)
from ct25d.transforms import RandomAffine2D, TargetStandardizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path, help="CSV with one case per row")
    p.add_argument("target", help="column to regress")
    p.add_argument("model_out", type=Path, help="checkpoint path to write")

    g = p.add_argument_group("columns")
    g.add_argument("--image-col", default="image")
    g.add_argument("--mask-col", default="mask")
    g.add_argument("--group-col", default=None,
                   help="split over these groups (e.g. patient_id) to avoid leakage")
    g.add_argument("--split-col", default=None,
                   help="use this train/val/test column instead of a random split")
    g.add_argument("--val-frac", type=float, default=0.2)
    g.add_argument("--test-frac", type=float, default=0.0,
                   help="hold out this fraction as a test split, reported once "
                        "after training and never used for selection")
    g.add_argument("--split-out", type=Path, default=None,
                   help="write the split actually used (one row per case) here")
    g.add_argument("--encoding", default=None,
                   help="CSV encoding; default tries UTF-8 (with or without BOM) "
                        "then CP932")
    g.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                   help="rewrite a path prefix in the image and mask columns, "
                        "e.g. /home/hanaoka=/mnt/w; may be repeated")

    g = p.add_argument_group("preprocessing")
    g.add_argument("--crop-size", type=int, default=96,
                   help="patch side in resampled pixels; 96 px at 0.78125 mm is "
                        "a 75 mm field of view, so larger structures are clipped")
    g.add_argument("--allow-clipped", action="store_true",
                   help="train anyway when structures do not fit in the patch")
    g.add_argument("--n-slices", type=int, default=3)
    g.add_argument("--gap-mm", type=float, default=SLICE_GAP_MM)
    g.add_argument("--slab-mm", type=float, default=SLAB_MM,
                   help="average each plane over this thickness, so thin-slice "
                        "volumes match 5 mm ones; 0 takes a single plane")
    g.add_argument("--in-plane-mm", type=float, default=TARGET_INPLANE_MM)
    g.add_argument("--label-value", type=int, default=1,
                   help="value of the target structure in a multi-label file")
    g.add_argument("--window", type=float, nargs=2, default=list(CT_WINDOW),
                   metavar=("LO", "HI"))
    g.add_argument("--cache", type=Path, default=None,
                   help="npz file to read/write the prepared stacks")

    g = p.add_argument_group("gating")
    g.add_argument("--gate-radius-mm", type=float, default=10.0,
                   help="0 disables gating")
    g.add_argument("--gate-profile", default="cosine",
                   choices=["cosine", "linear", "smoothstep", "gaussian"])
    g.add_argument("--gate-floor", type=float, default=0.0)
    g.add_argument("--gate-3d", action="store_true",
                   help="spherical gate: attenuate the neighbouring slices too")
    g.add_argument("--mask-channel", default="sdf",
                   choices=["sdf", "binary", "gate", "none"])
    g.add_argument("--keep-context", action="store_true",
                   help="add the un-gated centre slice as an extra channel")

    g = p.add_argument_group("target")
    g.add_argument("--task", default="regression",
                   choices=["regression", "classification"])
    g.add_argument("--n-classes", type=int, default=0,
                   help="number of classes for --task classification "
                        "(default: max label + 1)")
    g.add_argument("--label-smoothing", type=float, default=0.0)
    g.add_argument("--log-target", action="store_true",
                   help="standardize log1p(y); for positive, right-skewed targets")
    g.add_argument("--target-power", type=int, default=0, choices=[0, 1, 2],
                   help="0 scale-invariant, 1 length-like, 2 area-like")

    g = p.add_argument_group("augmentation")
    g.add_argument("--translate", type=float, default=0.08)
    g.add_argument("--scale", type=float, nargs=2, default=[0.9, 1.1])
    g.add_argument("--rotate-deg", type=float, default=5.0)
    g.add_argument("--shear", type=float, default=0.03)
    g.add_argument("--hflip", action="store_true")
    g.add_argument("--z-flip", action="store_true")
    g.add_argument("--no-augment", action="store_true")
    g.add_argument("--hu-shift", type=float, default=0.0,
                   help="add one uniform offset in [-x, +x] HU per case to the "
                        "whole image before windowing (training only)")

    g = p.add_argument_group("model and optimization")
    g.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet34"])
    g.add_argument("--norm", default="group",
                   choices=["group", "batch", "instance", "none"])
    g.add_argument("--dropout", type=float, default=0.2)
    g.add_argument("--no-cbam", action="store_true")
    g.add_argument("--epochs", type=int, default=60)
    g.add_argument("--batch-size", type=int, default=16)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-epochs", type=int, default=5)
    g.add_argument("--ramp-epochs", type=int, default=5)
    g.add_argument("--beta-nll", type=float, default=0.5)
    g.add_argument("--patience", type=int, default=20,
                   help="stop after this many epochs without a better val NLL")
    g.add_argument("--num-workers", type=int, default=0)
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    g.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def load_or_prepare(df, args):
    """Resampling dominates the wall clock, so cache it."""
    prep = dict(image_col=args.image_col, mask_col=args.mask_col,
                crop_size=args.crop_size, n_slices=args.n_slices,
                gap_mm=args.gap_mm, slab_mm=args.slab_mm,
                in_plane_mm=args.in_plane_mm, label_value=args.label_value)
    if args.cache is not None and args.cache.exists():
        try:
            stacks, kept = load_stack_cache(args.cache, prep)
        except ValueError as err:
            sys.exit(f"[error] {err}")
        print(f"loaded cached stacks from {args.cache}")
        return stacks, kept

    print(f"preparing {len(df)} cases ...")
    t0 = time.time()
    stacks, kept, _ = prepare_stacks(df, **prep)
    print(f"  {len(kept)}/{len(df)} usable, {time.time() - t0:.0f}s, "
          f"stack {stacks.shape[1:]}")
    if args.cache is not None:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        save_stack_cache(args.cache, stacks, kept, prep)
        print(f"  cached to {args.cache}")
    return stacks, kept


def check_field_of_view(stacks, args):
    """
    A structure wider than the patch is clipped at the border, and nothing
    downstream notices: the mask channel simply stops, and the model is trained
    on a truncated structure against the full-size label. Detect it here, where
    the mask for every case is already in memory.
    """
    mask = stacks[:, -1] > 0
    touches = (mask[:, 0, :].any(1) | mask[:, -1, :].any(1)
               | mask[:, :, 0].any(1) | mask[:, :, -1].any(1))
    n = int(touches.sum())
    fov = args.crop_size * args.in_plane_mm
    if n == 0:
        print(f"field of view: {args.crop_size} px = {fov:.0f} mm, "
              f"every structure fits")
        return

    # how wide the surviving part is, as a lower bound on the true extent
    widths = []
    for i in np.flatnonzero(touches):
        yy, xx = np.nonzero(mask[i])
        widths.append(max(yy.max() - yy.min(), xx.max() - xx.min())
                      * args.in_plane_mm)
    need = required_patch_size(float(np.max(widths)), args.in_plane_mm,
                               gate_radius_mm=max(args.gate_radius_mm, 0.0),
                               scale_max=max(args.scale), rotate_deg=args.rotate_deg,
                               translate=args.translate)
    msg = (f"{n}/{len(stacks)} structures reach the edge of the "
           f"{args.crop_size} px ({fov:.0f} mm) patch and are being clipped. "
           f"The visible part is already {np.max(widths):.0f} mm wide, so the "
           f"true extent is larger. Try --crop-size {need} or more; run "
           f"crop.py to see the required size for the whole cohort.")
    if args.allow_clipped:
        print(f"[warn] {msg}")
    else:
        sys.exit(f"[error] {msg}\n"
                 f"        Pass --allow-clipped to train anyway.")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mus, sigmas, zs = [], [], []
    for x, z in loader:
        mu, log_var = model(x.to(device))
        mus.append(mu.cpu().view(-1))
        sigmas.append(torch.exp(0.5 * log_var).cpu().view(-1))
        zs.append(z.view(-1))
    return (torch.cat(mus).numpy(), torch.cat(sigmas).numpy(),
            torch.cat(zs).numpy())


@torch.no_grad()
def evaluate_logits(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y.view(-1))
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def softmax_np(logits, temperature=1.0):
    z = logits / float(temperature)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def load_table(args):
    """The CSV with paths remapped and the target coerced to numbers."""
    import pandas as pd
    df = read_table(args.csv, encoding=args.encoding)
    if args.target not in df.columns:
        sys.exit(f"target column {args.target!r} not in the CSV "
                 f"(columns: {list(df.columns)})")
    df = apply_path_map(df, [args.image_col, args.mask_col],
                        parse_path_map(args.path_map))
    # spreadsheet placeholders such as #N/A or FALSE become NaN and are dropped
    df[args.target] = pd.to_numeric(df[args.target], errors="coerce")
    n0 = len(df)
    df = df[df[args.target].notna()].reset_index(drop=True)
    print(f"{args.csv}: {len(df)} rows with a value for {args.target!r}"
          + (f" ({n0 - len(df)} without)" if n0 > len(df) else ""))
    return df


def write_split(path, df, parts, args):
    which = np.full(len(df), "", dtype=object)
    for name, idx in parts.items():
        which[idx] = name
    cols = [c for c in (args.group_col, args.image_col, args.mask_col)
            if c and c in df.columns]
    out = df[cols].copy()
    out["split"] = which
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote the split to {path}")


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    classify = args.task == "classification"

    df = load_table(args)
    stacks, kept = load_or_prepare(df, args)
    df = df.iloc[kept].reset_index(drop=True)
    targets = df[args.target].to_numpy(dtype=np.float64)

    check_field_of_view(stacks, args)

    tr, va, te = split_three(df, val_frac=args.val_frac, test_frac=args.test_frac,
                             group_col=args.group_col, split_col=args.split_col,
                             seed=args.seed)
    print(f"split: {len(tr)} train / {len(va)} val / {len(te)} test"
          + (f" (grouped by {args.group_col})" if args.group_col else ""))
    if args.split_out is not None:
        write_split(args.split_out, df, {"train": tr, "val": va, "test": te}, args)

    if classify:
        if np.any(targets != np.round(targets)) or targets.min() < 0:
            sys.exit(f"[error] --task classification needs integer labels >= 0 "
                     f"in {args.target!r}")
        n_classes = args.n_classes or int(targets.max()) + 1
        if targets.max() >= n_classes:
            sys.exit(f"[error] label {int(targets.max())} >= --n-classes {n_classes}")
        counts = np.bincount(targets[tr].astype(int), minlength=n_classes)
        print(f"classes {n_classes}, train counts {counts.tolist()}")
        std = None
    else:
        n_classes = 0
        print(f"target {args.target}: {targets.min():.4g} .. {targets.max():.4g}, "
              f"median {np.median(targets):.4g}")
        # Fitted on the training split only; a standardizer that has seen the
        # validation targets leaks their distribution into every prediction.
        std = TargetStandardizer(log_transform=args.log_target).fit(targets[tr])

    gate = None if args.gate_radius_mm <= 0 else DistanceGate(
        radius_mm=args.gate_radius_mm, pixel_mm=args.in_plane_mm,
        profile=args.gate_profile, floor=args.gate_floor,
        z_offsets_mm=(None if not args.gate_3d else
                      tuple((np.arange(args.n_slices)
                             - (args.n_slices - 1) / 2) * args.gap_mm)))

    aug = None if args.no_augment else RandomAffine2D(
        translate=args.translate, scale=tuple(args.scale),
        rotate_deg=args.rotate_deg, shear=args.shear, hflip=args.hflip,
        z_flip=args.z_flip,
        image_channels=tuple(range(args.n_slices)),
        mask_channels=(args.n_slices,))

    common = dict(standardizer=std, gate=gate, window=tuple(args.window),
                  mask_channel=args.mask_channel, keep_context=args.keep_context,
                  task=args.task)
    train_ds = SliceStackDataset(stacks[tr], targets[tr], augment=aug,
                                 target_scale_power=args.target_power,
                                 hu_shift=args.hu_shift, **common)
    val_ds = SliceStackDataset(stacks[va], targets[va], augment=None,
                               target_scale_power=0, **common)
    test_ds = (SliceStackDataset(stacks[te], targets[te], augment=None,
                                 target_scale_power=0, **common)
               if len(te) else None)

    arch = dict(name=args.arch, n_slices=args.n_slices,
                n_mask_channels=train_ds.n_channels - args.n_slices,
                norm=args.norm, dropout=args.dropout, use_cbam=not args.no_cbam)
    if classify:
        arch["n_classes"] = n_classes
    model = build_model(arch).to(args.device)
    print(f"{args.arch}: {train_ds.n_channels} input channels, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"device {args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size,
                              num_workers=args.num_workers,
                              persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)
    test_loader = (None if test_ds is None else
                   DataLoader(test_ds, batch_size=args.batch_size,
                              num_workers=args.num_workers))

    config = dict(
        arch=arch, target=args.target, task=args.task,
        image_col=args.image_col, mask_col=args.mask_col,
        crop_size=args.crop_size, n_slices=args.n_slices, gap_mm=args.gap_mm,
        slab_mm=args.slab_mm, in_plane_mm=args.in_plane_mm,
        label_value=args.label_value, window=list(args.window),
        mask_channel=args.mask_channel,
        keep_context=bool(args.keep_context),
        gate=(None if gate is None else dict(
            radius_mm=args.gate_radius_mm, pixel_mm=args.in_plane_mm,
            profile=args.gate_profile, floor=args.gate_floor,
            z_offsets_mm=(list(gate.z_offsets_mm) if gate.z_offsets_mm else None))),
        log_target=bool(args.log_target), target_power=args.target_power,
        hu_shift=args.hu_shift, n_classes=n_classes,
    )

    loaders = dict(train=train_loader, val=val_loader, test=test_loader)
    if classify:
        train_classifier(model, opt, sched, loaders, config, args, n_classes,
                         sizes=(len(tr), len(va), len(te)))
    else:
        train_regressor(model, opt, sched, loaders, std, config, args, targets,
                        va, te, sizes=(len(tr), len(va), len(te)))


def train_regressor(model, opt, sched, loaders, std, config, args, targets,
                    va, te, sizes):
    crit = WarmupHeteroscedasticLoss(warmup_epochs=args.warmup_epochs,
                                     ramp_epochs=args.ramp_epochs,
                                     beta=args.beta_nll)
    train_loader, val_loader = loaders["train"], loaders["val"]
    unit = args.target
    space = "log-standardized" if args.log_target else "standardized"
    print(f"val_mae is in units of {unit}; the bracketed value is the same error "
          f"in the {space} space the loss works in")
    best = {"nll": float("inf"), "epoch": -1}
    best_state = None
    settle = args.warmup_epochs + args.ramp_epochs   # NLL is only the objective here
    for epoch in range(args.epochs):
        model.train()
        total = n = 0
        for x, z in train_loader:
            x, z = x.to(args.device), z.to(args.device)
            mu, log_var = model(x)
            loss = crit(mu, log_var, z, epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.size(0)
            n += x.size(0)
        sched.step()

        mu_z, sigma_z, z_true = evaluate(model, val_loader, args.device)
        rep = uncertainty_report(mu_z, sigma_z, z_true)
        # the same predictions in the units of the target, so the number on
        # screen is one that can be compared with clinical expectations
        rep_y = uncertainty_report(std.inverse_transform(mu_z),
                                   std.inverse_transform(mu_z + sigma_z)
                                   - std.inverse_transform(mu_z),
                                   std.inverse_transform(z_true))
        mark = ""
        if epoch >= settle and rep["nll"] < best["nll"]:
            best = {"nll": rep["nll"], "epoch": epoch, "z_std": rep["z_std"],
                    "mae_z": rep["mae"], "mae": rep_y["mae"]}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = "  *"
        print(f"epoch {epoch:3d}  alpha {crit.alpha(epoch):.2f}  "
              f"train {total / n:8.4f}  val_nll {rep['nll']:7.4f}  "
              f"val_mae {rep_y['mae']:9.4g} {unit}  "
              f"({rep['mae']:.3f} sd)  z_std {rep['z_std']:.3f}{mark}",
              flush=True)

        if best["epoch"] >= 0 and epoch - best["epoch"] >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    if best_state is None:
        print("[warn] training ended before the NLL phase; keeping the last weights")
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(args.device)

    # Calibrate sigma on the validation split, in standardized space -- a scalar
    # factor there applies unchanged after the inverse transform.
    mu_z, sigma_z, z_true = evaluate(model, val_loader, args.device)
    scale = fit_sigma_scale(mu_z, sigma_z, z_true)

    def report(mu_z, sigma_z, z_true, y):
        mean, sigma = std.inverse_transform(mu_z, sigma_z * scale)
        out = uncertainty_report(mean, sigma, y)
        # calibration figures in the standardized space, where the intervals
        # that eval.py writes are built (see TargetStandardizer.interval)
        out_z = uncertainty_report(mu_z, sigma_z * scale, z_true)
        for k in ("z_std", "coverage_95"):
            out[k] = out_z[k]
        return out

    final = report(mu_z, sigma_z, z_true, targets[va])
    print(f"\nbest epoch {best['epoch']}, sigma scale {scale:.3f}")
    print("validation, in the units of " + args.target
          + " (z_std and coverage_95 in the standardized space):")
    for k, v in final.items():
        print(f"  {k:14s} {v:.4f}")
    metrics = {"best_epoch": best["epoch"], **final}
    if loaders["test"] is not None:
        test = report(*evaluate(model, loaders["test"], args.device), targets[te])
        print("test (held out, reported once):")
        for k, v in test.items():
            print(f"  {k:14s} {v:.4f}")
        metrics.update({f"test_{k}": v for k, v in test.items()})

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.model_out, model, std, config, sigma_scale=scale,
                    metrics=metrics)
    print(f"\nwrote {args.model_out}")
    print(json.dumps({"target": args.target, "n_train": sizes[0],
                      "n_val": sizes[1], "n_test": sizes[2],
                      "sigma_scale": round(scale, 4),
                      "val_mae": round(final["mae"], 4),
                      **({"test_mae": round(metrics["test_mae"], 4)}
                         if "test_mae" in metrics else {})}, indent=None))


def train_classifier(model, opt, sched, loaders, config, args, n_classes, sizes):
    crit = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    best = {"nll": float("inf"), "epoch": -1}
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = n = 0
        for x, y in loaders["train"]:
            x, y = x.to(args.device), y.to(args.device)
            loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * x.size(0)
            n += x.size(0)
        sched.step()

        logits, y_val = evaluate_logits(model, loaders["val"], args.device)
        rep = classification_report(softmax_np(logits), y_val)
        mark = ""
        if rep["nll"] < best["nll"]:
            best = {"nll": rep["nll"], "epoch": epoch, "accuracy": rep["accuracy"]}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = "  *"
        print(f"epoch {epoch:3d}  train {total / n:8.4f}  val_nll {rep['nll']:7.4f}  "
              f"val_acc {rep['accuracy']:.3f}  ece {rep['ece']:.3f}{mark}",
              flush=True)
        if epoch - best["epoch"] >= args.patience:
            print(f"no improvement for {args.patience} epochs, stopping")
            break

    model.load_state_dict(best_state)
    model.to(args.device)

    logits, y_val = evaluate_logits(model, loaders["val"], args.device)
    temperature = fit_temperature(logits, y_val)
    final = classification_report(softmax_np(logits, temperature), y_val)
    raw = classification_report(softmax_np(logits), y_val)
    print(f"\nbest epoch {best['epoch']}, temperature {temperature:.3f} "
          f"(val nll {raw['nll']:.4f} -> {final['nll']:.4f}, "
          f"ece {raw['ece']:.4f} -> {final['ece']:.4f})")

    def show(name, rep):
        print(f"{name}:")
        for k, v in rep.items():
            if k != "confusion":
                print(f"  {k:14s} {v:.4f}")
        print("  confusion (rows truth, columns prediction):")
        for row in rep["confusion"]:
            print("    " + " ".join(f"{c:5d}" for c in row))

    show("validation", final)
    metrics = {"best_epoch": best["epoch"],
               **{k: v for k, v in final.items() if k != "confusion"}}
    if loaders["test"] is not None:
        logits_t, y_t = evaluate_logits(model, loaders["test"], args.device)
        test = classification_report(softmax_np(logits_t, temperature), y_t)
        show("test (held out, reported once)", test)
        metrics.update({f"test_{k}": v for k, v in test.items() if k != "confusion"})

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.model_out, model, None, config, metrics=metrics,
                    temperature=temperature)
    print(f"\nwrote {args.model_out}")
    print(json.dumps({"target": args.target, "n_train": sizes[0],
                      "n_val": sizes[1], "n_test": sizes[2],
                      "temperature": round(temperature, 4),
                      "val_accuracy": round(final["accuracy"], 4),
                      **({"test_accuracy": round(metrics["test_accuracy"], 4)}
                         if "test_accuracy" in metrics else {})}, indent=None))


if __name__ == "__main__":
    main()
