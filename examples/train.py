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

from ct25d.calibration import fit_sigma_scale, uncertainty_report
from ct25d.checkpoint import build_model, save_checkpoint
from ct25d.constants import CT_WINDOW, SLICE_GAP_MM, TARGET_INPLANE_MM
from ct25d.data import SliceStackDataset
from ct25d.gating import DistanceGate
from ct25d.geometry import required_patch_size
from ct25d.losses import WarmupHeteroscedasticLoss
from ct25d.tabular import prepare_stacks, split_indices
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
                   help="use this train/val column instead of a random split")
    g.add_argument("--val-frac", type=float, default=0.2)

    g = p.add_argument_group("preprocessing")
    g.add_argument("--crop-size", type=int, default=96,
                   help="patch side in resampled pixels; 96 px at 0.78125 mm is "
                        "a 75 mm field of view, so larger structures are clipped")
    g.add_argument("--allow-clipped", action="store_true",
                   help="train anyway when structures do not fit in the patch")
    g.add_argument("--n-slices", type=int, default=3)
    g.add_argument("--gap-mm", type=float, default=SLICE_GAP_MM)
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
    if args.cache is not None and args.cache.exists():
        z = np.load(args.cache)
        print(f"loaded cached stacks from {args.cache}")
        return z["stacks"], z["kept"]

    print(f"preparing {len(df)} cases ...")
    t0 = time.time()
    stacks, kept, _ = prepare_stacks(
        df, image_col=args.image_col, mask_col=args.mask_col,
        crop_size=args.crop_size, n_slices=args.n_slices, gap_mm=args.gap_mm,
        in_plane_mm=args.in_plane_mm, label_value=args.label_value)
    print(f"  {len(kept)}/{len(df)} usable, {time.time() - t0:.0f}s, "
          f"stack {stacks.shape[1:]}")
    if args.cache is not None:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.cache, stacks=stacks, kept=kept)
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


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    import pandas as pd
    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        sys.exit(f"target column {args.target!r} not in the CSV "
                 f"(columns: {list(df.columns)})")
    df = df[df[args.target].notna()].reset_index(drop=True)
    print(f"{args.csv}: {len(df)} rows with a value for {args.target!r}")

    stacks, kept = load_or_prepare(df, args)
    df = df.iloc[kept].reset_index(drop=True)
    targets = df[args.target].to_numpy(dtype=np.float64)

    check_field_of_view(stacks, args)

    tr, va = split_indices(df, val_frac=args.val_frac, group_col=args.group_col,
                           split_col=args.split_col, seed=args.seed)
    print(f"split: {len(tr)} train / {len(va)} val"
          + (f" (grouped by {args.group_col})" if args.group_col else ""))
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
                  mask_channel=args.mask_channel, keep_context=args.keep_context)
    train_ds = SliceStackDataset(stacks[tr], targets[tr], augment=aug,
                                 target_scale_power=args.target_power, **common)
    val_ds = SliceStackDataset(stacks[va], targets[va], augment=None,
                               target_scale_power=0, **common)

    arch = dict(name=args.arch, n_slices=args.n_slices,
                n_mask_channels=train_ds.n_channels - args.n_slices,
                norm=args.norm, dropout=args.dropout, use_cbam=not args.no_cbam)
    model = build_model(arch).to(args.device)
    print(f"{args.arch}: {train_ds.n_channels} input channels, "
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
          f"device {args.device}")

    crit = WarmupHeteroscedasticLoss(warmup_epochs=args.warmup_epochs,
                                     ramp_epochs=args.ramp_epochs,
                                     beta=args.beta_nll)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size,
                              num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            num_workers=args.num_workers)

    config = dict(
        arch=arch, target=args.target,
        image_col=args.image_col, mask_col=args.mask_col,
        crop_size=args.crop_size, n_slices=args.n_slices, gap_mm=args.gap_mm,
        in_plane_mm=args.in_plane_mm, label_value=args.label_value,
        window=list(args.window), mask_channel=args.mask_channel,
        keep_context=bool(args.keep_context),
        gate=(None if gate is None else dict(
            radius_mm=args.gate_radius_mm, pixel_mm=args.in_plane_mm,
            profile=args.gate_profile, floor=args.gate_floor,
            z_offsets_mm=(list(gate.z_offsets_mm) if gate.z_offsets_mm else None))),
        log_target=bool(args.log_target), target_power=args.target_power,
    )

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
        mark = ""
        if epoch >= settle and rep["nll"] < best["nll"]:
            best = {"nll": rep["nll"], "epoch": epoch, "z_std": rep["z_std"],
                    "mae_z": rep["mae"]}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = "  *"
        print(f"epoch {epoch:3d}  alpha {crit.alpha(epoch):.2f}  "
              f"train {total / n:8.4f}  val_nll {rep['nll']:7.4f}  "
              f"val_mae_z {rep['mae']:.4f}  z_std {rep['z_std']:.3f}{mark}",
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
    mean, sigma = std.inverse_transform(mu_z, sigma_z * scale)
    final = uncertainty_report(mean, sigma, targets[va])
    print(f"\nbest epoch {best['epoch']}, sigma scale {scale:.3f}")
    print("validation, in the units of " + args.target + ":")
    for k, v in final.items():
        print(f"  {k:14s} {v:.4f}")

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(args.model_out, model, std, config, sigma_scale=scale,
                    metrics={"best_epoch": best["epoch"], **final})
    print(f"\nwrote {args.model_out}")
    print(json.dumps({"target": args.target, "n_train": len(tr), "n_val": len(va),
                      "sigma_scale": round(scale, 4),
                      "val_mae": round(final["mae"], 4)}, indent=None))


if __name__ == "__main__":
    main()
