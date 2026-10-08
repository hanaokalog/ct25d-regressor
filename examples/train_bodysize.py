#!/usr/bin/env python
"""Train the body-size (height, weight) model on prepared cases.

    python examples/train_bodysize.py cases.csv bodysize.pt --split-col split

The CSV has one row per case: the path of a prepared case (an .npz written from
ct25d.bodysize.prepare_case), the targets and a train/val/test split column.
Rows with a missing target are left out of training. The checkpoint carries the
weights, the input geometry, a standardizer and a calibrated sigma scale per
target.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ct25d.bodysize import (
    BodySizeDataset,
    BodySizeNet,
    SingleLevelDataset,
    load_case,
    save_bodysize,
)
from ct25d.calibration import fit_sigma_scale, uncertainty_report
from ct25d.losses import WarmupHeteroscedasticLoss
from ct25d.tabular import read_table, split_three
from ct25d.transforms import TargetStandardizer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", type=Path)
    p.add_argument("model_out", type=Path)
    p.add_argument("--case-col", default="case")
    p.add_argument("--targets", nargs="+", default=["height", "weight"])
    p.add_argument("--split-col", default=None)
    p.add_argument("--group-col", default=None)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--test-frac", type=float, default=0.15)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--p-drop", type=float, default=0.0,
                   help="probability each of a training scan that misses L1 or L3 "
                        "(its planes emptied, the scan cut between the levels); "
                        "> 0 makes the model usable with one level")
    p.add_argument("--norm", default="group",
                   choices=["group", "batch", "instance", "none"])
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--no-cbam", action="store_true")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--ramp-epochs", type=int, default=5)
    p.add_argument("--beta-nll", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


@torch.no_grad()
def predict_z(model, loader, device):
    model.eval()
    mus, sds, ys = [], [], []
    for proj, axial, y in loader:
        mu, log_var = model(proj.to(device), axial.to(device))
        mus.append(mu.float().cpu())
        sds.append(torch.exp(0.5 * log_var).float().cpu())
        ys.append(y)
    return torch.cat(mus).numpy(), torch.cat(sds).numpy(), torch.cat(ys).numpy()


def report(name, title, mu_z, sd_z, scales, stds, y, targets):
    out = {}
    print(f"{title}:")
    for k, t in enumerate(targets):
        mean, sigma = stds[t].inverse_transform(mu_z[:, k], sd_z[:, k] * scales[t])
        r = uncertainty_report(mean, sigma, y[:, k])
        rz = uncertainty_report(mu_z[:, k], sd_z[:, k] * scales[t],
                                stds[t].transform(y[:, k]))
        r["z_std"], r["coverage_95"] = rz["z_std"], rz["coverage_95"]
        shown = ("n", "mae", "rmse", "bias", "z_std", "coverage_95")
        print(f"  {t:8s} " + "  ".join(f"{m} {r[m]:.3f}" for m in shown))
        out.update({f"{name}_{t}_{m}": float(r[m]) for m in shown[1:]})
    return out


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = read_table(args.csv)
    import pandas as pd
    for t in args.targets:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    n0 = len(df)
    df = df[df[args.targets].notna().all(1)].reset_index(drop=True)
    print(f"{args.csv}: {len(df)} cases with every target ({n0 - len(df)} without)")
    tr, va, te = split_three(df, val_frac=args.val_frac, test_frac=args.test_frac,
                             group_col=args.group_col, split_col=args.split_col,
                             seed=args.seed)
    print(f"split: {len(tr)} train / {len(va)} val / {len(te)} test")

    t0 = time.time()
    cases = [load_case(p) for p in df[args.case_col]]
    geometry = json.loads(str(np.load(df[args.case_col].iloc[0])["geometry"]))
    geometry.setdefault("l1_l3_mm", 65.0)     # cases prepared before it was recorded
    l1_l3_rows = geometry["l1_l3_mm"] / geometry["proj_mm"]
    print(f"loaded {len(cases)} prepared cases in {time.time() - t0:.0f}s")

    y = df[args.targets].to_numpy(np.float64)
    stds = {t: TargetStandardizer().fit(y[tr, k]) for k, t in enumerate(args.targets)}
    z = np.stack([stds[t].transform(y[:, k]) for k, t in enumerate(args.targets)], 1)

    def subset(idx, augment):
        return BodySizeDataset([cases[i] for i in idx], z[idx], augment=augment,
                               p_drop=args.p_drop, l1_l3_rows=l1_l3_rows)
    train_ds = subset(tr, not args.no_augment)
    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    # workers are re-created every epoch (no persistent_workers) so that
    # train_ds.set_epoch reaches them
    val_loader = DataLoader(subset(va, False), **loader_kw)
    test_loader = DataLoader(subset(te, False), **loader_kw) if len(te) else None

    model_cfg = dict(norm=args.norm, dropout=args.dropout, use_cbam=not args.no_cbam,
                     n_targets=len(args.targets))
    model = BodySizeNet(**model_cfg).to(args.device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"BodySizeNet: {n_params:.2f}M params, device {args.device}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = WarmupHeteroscedasticLoss(warmup_epochs=args.warmup_epochs,
                                     ramp_epochs=args.ramp_epochs, beta=args.beta_nll)
    settle = args.warmup_epochs + args.ramp_epochs
    best, best_state = {"nll": float("inf"), "epoch": -1}, None
    for epoch in range(args.epochs):
        train_ds.set_epoch(epoch)
        model.train()
        total = n = 0
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kw)
        for proj, axial, zt in train_loader:
            proj, axial, zt = (v.to(args.device) for v in (proj, axial, zt))
            mu, log_var = model(proj, axial)
            loss = sum(crit(mu[:, k], log_var[:, k], zt[:, k], epoch)
                       for k in range(len(args.targets)))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) * len(zt)
            n += len(zt)
        sched.step()

        mu_z, sd_z, zv = predict_z(model, val_loader, args.device)
        nll = float(np.mean(0.5 * (2 * np.log(sd_z) + (zv - mu_z) ** 2 / sd_z ** 2)))
        maes = [float(np.mean(np.abs(stds[t].inverse_transform(mu_z[:, k])
                                     - stds[t].inverse_transform(zv[:, k]))))
                for k, t in enumerate(args.targets)]
        mark = ""
        if epoch >= settle and nll < best["nll"]:
            best = {"nll": nll, "epoch": epoch}
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        print(f"epoch {epoch:3d}  train {total / max(n, 1):8.4f}  val_nll {nll:7.4f}  "
              + "  ".join(f"mae_{t} {m:.2f}" for t, m in zip(args.targets, maes))
              + mark,
              flush=True)
        if best["epoch"] >= 0 and epoch - best["epoch"] >= args.patience:
            print(f"no better val NLL for {args.patience} epochs, stopping")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("[warn] training ended before the NLL phase; keeping the last weights")
    mu_z, sd_z, _ = predict_z(model, val_loader, args.device)
    scales = {t: fit_sigma_scale(mu_z[:, k], sd_z[:, k], z[va, k])
              for k, t in enumerate(args.targets)}
    print(f"best epoch {best['epoch']}, sigma scales "
          + ", ".join(f"{t} {s:.3f}" for t, s in scales.items()))
    metrics = report("val", "validation", mu_z, sd_z, scales, stds, y[va], args.targets)
    if test_loader is not None:
        mu_t, sd_t, _ = predict_z(model, test_loader, args.device)
        metrics.update(report("test", "test (held out, reported once)", mu_t, sd_t,
                              scales, stds, y[te], args.targets))
        if args.p_drop > 0:
            for kept in ("L1", "L3"):
                one = SingleLevelDataset([cases[i] for i in te], z[te], kept,
                                         l1_l3_rows)
                mu_k, sd_k, _ = predict_z(model, DataLoader(one, **loader_kw),
                                          args.device)
                title = f"test, {kept} only (the scan cut halfway between L1 and L3)"
                metrics.update(report(f"test_{kept}_only", title, mu_k, sd_k,
                                      scales, stds, y[te], args.targets))
    config = dict(model=model_cfg, targets=list(args.targets), geometry=geometry,
                  best_epoch=best["epoch"], p_drop=args.p_drop)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_bodysize(args.model_out, model, stds, config, scales, metrics)
    print(f"wrote {args.model_out}")


if __name__ == "__main__":
    main()
