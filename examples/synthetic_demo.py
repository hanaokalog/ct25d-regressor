"""End-to-end run on synthetic data: no patient images needed.

    python examples/synthetic_demo.py

Builds phantom volumes with a disc of known area, runs the full pipeline, and
prints the calibration report. The point is to exercise every stage and to show
the order the pieces go in, not to demonstrate accuracy on a real task.
"""

import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import DataLoader

from ct25d.calibration import uncertainty_report
from ct25d.data import SliceStackDataset
from ct25d.gating import DistanceGate
from ct25d.geometry import build_sample_sitk, mask_area_mm2
from ct25d.losses import WarmupHeteroscedasticLoss
from ct25d.models import resnet18_cbam25d
from ct25d.transforms import RandomAffine2D, TargetStandardizer

N_CASES = 96
EPOCHS = 12
CROP = 96


def make_phantom(rng):
    """A volume with a bright disc on the centre slice, plus a distractor."""
    W = H = 180
    Z = 12
    inplane, z_spacing = 0.9766, 2.5
    cz = Z // 2
    r = rng.uniform(5, 20)

    yy, xx = np.mgrid[0:H, 0:W]
    vol = rng.normal(-30, 25, (Z, H, W)).astype(np.float32)
    for z in range(Z):
        shrink = max(0.0, 1.0 - abs(z - cz) / 8.0)
        vol[z] += 700.0 * ((yy - 90) ** 2 + (xx - 90) ** 2 <= (r * shrink) ** 2)
        # a distractor of a different size, far enough away to be gated out
        vol[z] += 700.0 * ((yy - 30) ** 2 + (xx - 140) ** 2 <= (2 * r) ** 2)
    mask = ((yy - 90) ** 2 + (xx - 90) ** 2 <= r ** 2).astype(np.uint8)

    img = sitk.GetImageFromArray(vol)
    lab = sitk.GetImageFromArray(np.stack([mask if z == cz else np.zeros_like(mask)
                                           for z in range(Z)]))
    for it in (img, lab):
        it.SetSpacing((inplane, inplane, z_spacing))
        it.SetOrigin((-90.0, -90.0, 0.0))
    return img, lab


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    stacks, targets = [], []
    for _ in range(N_CASES):
        img, lab = make_phantom(rng)
        stacks.append(build_sample_sitk(img, lab, crop_size=CROP))
        targets.append(mask_area_mm2(lab))          # from the ORIGINAL mask
    stacks = np.stack(stacks)
    targets = np.array(targets)
    print(f"{len(stacks)} cases, stack {stacks.shape[1:]}, "
          f"target {targets.min():.0f}-{targets.max():.0f} mm^2")

    n_val = N_CASES // 4
    tr, va = slice(n_val, None), slice(0, n_val)

    std = TargetStandardizer(log_transform=True).fit(targets[tr])   # train only
    gate = DistanceGate(radius_mm=10.0, profile="cosine")
    aug = RandomAffine2D(translate=0.08, scale=(0.9, 1.1), rotate_deg=5.0, shear=0.03)

    train_ds = SliceStackDataset(stacks[tr], targets[tr], std, augment=aug,
                                 gate=gate, target_scale_power=2,
                                 mask_channel="sdf", keep_context=True)
    val_ds = SliceStackDataset(stacks[va], targets[va], std, augment=None,
                               gate=gate, mask_channel="sdf", keep_context=True)

    model = resnet18_cbam25d(n_slices=3, n_mask_channels=2, norm="group")
    crit = WarmupHeteroscedasticLoss(warmup_epochs=4, ramp_epochs=4, beta=0.5)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=8, shuffle=True, drop_last=True)
    print(f"input channels {train_ds.n_channels}, "
          f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    for epoch in range(EPOCHS):
        model.train()
        total = n = 0
        for xb, zb in loader:
            mu, log_var = model(xb)
            loss = crit(mu, log_var, zb, epoch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * xb.size(0)
            n += xb.size(0)
        print(f"  epoch {epoch:2d}  alpha {crit.alpha(epoch):.2f}  "
              f"loss {total / n:.4f}")

    xb = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
    mean, sigma = model.predict(xb, std)
    rep = uncertainty_report(mean.view(-1), sigma.view(-1), targets[va])
    print("\nvalidation:")
    for k, v in rep.items():
        print(f"  {k:14s} {v:.3f}")
    print("\n(z_std near 1 and coverage_95 near 0.95 mean the uncertainty is "
          "calibrated;\n multiply sigma by sigma_scale if it is not.)")


if __name__ == "__main__":
    main()
