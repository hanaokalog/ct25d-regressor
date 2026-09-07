# ct25d-regressor

2.5D CT regression with mask-guided attention and predictive uncertainty.

Given a CT volume and a binary mask of a structure on one slice, the model
predicts a scalar quantity about that structure together with a per-case
uncertainty. Three slices 5 mm apart are stacked as channels, the image is
gated to a 1 cm neighbourhood of the structure, and a ResNet with CBAM
attention outputs a Gaussian mean and variance.

```
sitk.Image volume + mask
  └─ geometry.build_sample_sitk      (4, H, W) in HU, 0.78125 mm, 5 mm slice gap
  └─ transforms.RandomAffine2D       affine augmentation, still in HU
  └─ gating.make_input               distance gate, HU window, channel layout
  └─ models.ResNet25DCBAMRegressor   mu, log_var
  └─ losses.WarmupHeteroscedasticLoss  Huber → Gaussian NLL
  └─ transforms.TargetStandardizer   back to physical units
  └─ calibration.uncertainty_report  is the uncertainty trustworthy
```

## Install

```bash
pip install -e ".[dev]"
```

PyTorch is a dependency but not pinned to a build; on a CPU-only machine
install it from the CPU index first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Quickstart

```python
import numpy as np, torch, SimpleITK as sitk
from torch.utils.data import DataLoader
from ct25d.geometry import build_sample_sitk, mask_area_mm2
from ct25d.transforms import RandomAffine2D, TargetStandardizer
from ct25d.gating import DistanceGate
from ct25d.data import SliceStackDataset
from ct25d.models import resnet18_cbam25d
from ct25d.losses import WarmupHeteroscedasticLoss

# 1. volumes -> (4, H, W) HU stacks. Targets come from the ORIGINAL mask.
stacks = np.stack([build_sample_sitk(img, lab, crop_size=96)
                   for img, lab in cases])
targets = np.array([mask_area_mm2(lab) for _, lab in cases])   # mm^2

# 2. fit the target standardizer on the training split only
std = TargetStandardizer(log_transform=True).fit(targets[train_idx])

# 3. dataset
ds = SliceStackDataset(
    stacks[train_idx], targets[train_idx], std,
    augment=RandomAffine2D(translate=0.08, scale=(0.9, 1.1),
                           rotate_deg=5.0, shear=0.03),
    gate=DistanceGate(radius_mm=10.0, profile="cosine"),
    target_scale_power=2,          # the target is an area
    mask_channel="sdf", keep_context=True,
)

# 4. train
model = resnet18_cbam25d(n_slices=3, n_mask_channels=2, norm="group")
crit = WarmupHeteroscedasticLoss(warmup_epochs=5, ramp_epochs=5, beta=0.5)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

for epoch in range(n_epochs):
    for x, z in DataLoader(ds, batch_size=8, shuffle=True):
        mu, log_var = model(x)
        loss = crit(mu, log_var, z, epoch)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

# 5. predict in physical units
mean, sigma = model.predict(x, std)      # mm^2, with a per-case sigma
```

A complete runnable version on synthetic phantoms:

```bash
python examples/synthetic_demo.py
```

## Command line

Training reads a CSV with one case per row: a path to a grayscale volume, a
path to a label volume whose structure sits on a single slice, and any number
of variables. Voxel sizes and matrix sizes may differ between rows.

| ct | seg | patient_id | area_mm2 | age |
|---|---|---|---|---|
| /data/001_ct.nii.gz | /data/001_seg.nii.gz | p001 | 412.6 | 64 |
| /data/002_ct.nii.gz | /data/002_seg.nii.gz | p001 | 388.1 | 64 |

```bash
# optional but recommended first: crop the cohort down to the labels
python examples/crop.py cases.csv cropped.csv --out-dir /data/crops \
    --image-col ct --mask-col seg --for-crop-size 96 --jobs 8

python examples/train.py cropped.csv area_mm2 model.pt \
    --image-col image_crop --mask-col mask_crop --group-col patient_id \
    --crop-size 96 --target-power 2 --log-target --epochs 120

python examples/eval.py held_out.csv area_mm2 model.pt --out predictions.csv
```

The three positional arguments are the same in train.py and eval.py — CSV, target
column, checkpoint — with the model written by one and read by the other.

`--group-col` splits over groups rather than rows, so two studies from the same
patient cannot land on opposite sides of the split. `--cache` stores the
resampled stacks in an npz, which is worth setting: resampling dominates the
wall clock and is identical every epoch.

The checkpoint carries the weights, the full preprocessing configuration, the
fitted `TargetStandardizer` and the calibrated sigma scale. `eval.py` takes all
of these from the file rather than from flags, so evaluation cannot drift from
training; the only thing it takes from the command line is where the data is.
It reports the calibration diagnostics, warns when `z_std` has moved away from
1 on the new set, and writes per-case predictions with 95% intervals. If the
target column is missing it predicts anyway and skips the report.

### Cropping

`crop.py` reads the same CSV, writes a cropped `.nii.gz` pair per row, and
writes a new CSV with every original column plus `image_crop` and `mask_crop`.
Point `train.py` at that CSV with `--image-col image_crop --mask-col mask_crop`
and nothing else changes: the crop keeps its origin and direction cosines, so
`build_sample_sitk` produces the same array either way — which is what the test
suite asserts, mask channel bit-identical and image channels equal to float32
rounding.

On 512x512x300 volumes with a single-slice label, 115 MB per case becomes under
1 MB, a 130-200x reduction in voxels read.

The label is read in full, since that is where the bounding box comes from, but
it is binary and small once compressed. Only the corresponding block of the
image is materialized, through `ImageFileReader` with an extract region. For
`.nii.gz` the file still has to be decompressed sequentially — a gzip stream has
no random access — but the full array is never allocated.

Set the margin with `--for-crop-size`, which works out how far the augmented
patch can reach (patch half-width, widest zoom-out, largest rotation swing,
largest translation) and covers it. Cropping too tightly fails silently: the
patch picks up the fill value outside the crop and the model sees a black wedge
that appears only on augmented samples. `--for-crop-size 96` gives 52 mm.

## Preprocessing geometry

In-plane pixels are resampled to **0.78125 mm** and the three slices are taken
**5 mm** apart in physical space, from `constants.py`. Everything is resolved
through SimpleITK in physical coordinates, so oblique direction cosines,
differing origins, and a mask stored on its own grid all work.

Each plane is resampled independently rather than as one 3D grid. That allows
the z position to be clamped inside the volume, so a structure on the first or
last slice gets a duplicated neighbour instead of a plane of air.

**Slices are stacked as channels, not fed to a 3D convolution.** The anisotropy
is 5 / 0.78125 = 6.4; a 3×3×3 kernel would span 1.56 mm in plane against 10 mm
through plane. Channel stacking also lets the first convolution learn
z-position-specific weights, which is what we want, since only the centre slice
carries the mask.

## Distance gating

`DistanceGate` keeps the image inside the mask at full weight, fades it to zero
over the next 10 mm, and blanks everything beyond.

- **Gating happens in the windowed [0, 1] domain, never in HU.** Multiplying HU
  by a taper converges to 0 HU — water — so the faded ring would read as a
  plausible soft-tissue structure. In [0, 1] it converges to the window floor,
  which is background.
- **The default falloff is a raised cosine, not a linear ramp.** A linear taper
  has a slope discontinuity at both the mask boundary and the 10 mm radius, and
  convolution filters respond to both. The cosine profile is flat at both ends.
- Gating is destructive. `keep_context=True` adds the un-gated centre slice as
  one extra channel, and `floor=0.05` leaves a faint trace of the surroundings.
- The mask channel defaults to a clipped **signed distance** rather than a
  binary mask: more informative about boundary geometry, and smooth under
  interpolation.

## Uncertainty

The model predicts `mu` and `log_var`; the loss is the Gaussian NLL with two
practical modifications.

- **Huber warmup.** At initialization the fastest way to reduce the NLL is to
  inflate the variance, after which the mean head receives gradients scaled by
  1/var and stops learning. `WarmupHeteroscedasticLoss` fits the mean with
  Huber first, then blends into the NLL over a few epochs.
- **β-NLL** (Seitzer et al., ICLR 2022) stops the network from down-weighting
  hard samples by inflating their variance.
- The log-variance bound is exact in the interior and bends only in the last
  unit at each end, so a saturated variance head still has a gradient and can
  recover. A hard clamp cannot.

Training successfully is not the same as being calibrated. `calibration.py`
reports `z_std` (should be ≈ 1), 95% coverage, and `corr_abs_err`, which is the
one that tells you the uncertainty is informative per case rather than merely
correct on average. `fit_sigma_scale` gives the single multiplicative
correction, fit on the validation split.

## Targets and augmentation

Zoom augmentation changes the apparent size of the structure, so a
size-dependent label has to follow it — `target_scale_power` is `0` for
scale-invariant targets (mean HU, ratios, scores), `1` for lengths, `2` for
areas.

Derive the label from the **original** mask and spacing (`mask_area_mm2`), not
from the resampled one. Resampling a mask with bilinear interpolation and a 0.5
threshold does not conserve area exactly; measured error is under 1% when
upsampling but around 8% for a 7 mm structure resampled from 0.7 mm. The
resampled mask is an input cue, not the source of truth for the label.

## Tests

```bash
pytest
```

106 tests covering resampling geometry against known slice positions, flipped
direction cosines, border clamping, falloff continuity, the gating intensity
domain, exact affine rotation on non-square images, the standardizer round
trip, the loss schedule, and an end-to-end run of all three CLIs against real
`.nii.gz` files with mismatched voxel and matrix sizes, including the
equivalence of the pipeline output before and after cropping. Tests requiring
`torch`, `SimpleITK` or `pandas` skip cleanly if those are absent.

## Notes

- `.gitignore` excludes NIfTI, DICOM, MetaImage and NRRD files by default.
  Check `git status` before the first commit anyway.
- The URLs in `pyproject.toml` assume the GitHub account `hanaokalog`; correct
  them if the account name differs from the email local part.

## References

- Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV 2018
- Kendall & Gal, *What Uncertainties Do We Need in Bayesian Deep Learning?*, NeurIPS 2017
- Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation*, ICLR 2022
- Goyal et al., *Accurate, Large Minibatch SGD* (zero-initialized residual γ), 2017
