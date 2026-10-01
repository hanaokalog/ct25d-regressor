# ct25d-regressor

2.5D CT regression with mask-guided attention and predictive uncertainty.

Given a CT volume and a binary mask of a structure on one slice, the model
predicts a scalar quantity about that structure together with a per-case
uncertainty. Three 5 mm slabs 5 mm apart are stacked as channels, the image is
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
wall clock and is identical every epoch. The cache records the preprocessing
settings that wrote it, and a cache written with different ones (or by a
version before slab averaging) is refused rather than reused.

The checkpoint carries the weights, the full preprocessing configuration, the
fitted `TargetStandardizer` and the calibrated sigma scale. `eval.py` takes all
of these from the file rather than from flags, so evaluation cannot drift from
training; the only thing it takes from the command line is where the data is.
It reports the calibration diagnostics, warns when `z_std` has moved away from
1 on the new set, and writes per-case predictions with 95% intervals. If the
target column is missing it predicts anyway and skips the report.

### Reading the training log

```
val_mae is in units of area_mm2; the bracketed value is the same error in the
log-standardized space the loss works in
epoch   2  alpha 1.00  train 0.5013  val_nll 1.1876  val_mae 1655 area_mm2  (0.749 sd)  z_std 0.799  *
```

`val_mae` is in the target's own units, so it can be compared against what the
measurement is actually for; the bracketed figure is the same error in the
standardized space the loss operates in, which is what the NLL and `z_std`
refer to. `alpha` is the weight of the NLL term in the warmup schedule, and `*`
marks a new best epoch. Epoch selection uses the standardized NLL, which is
scale-free; it only starts once the warmup and ramp are over, since the NLL is
not the objective before that.

### Field of view

`--crop-size` is a fixed number of pixels at a fixed millimetre spacing, so the
field of view is fixed too: **96 px at 0.78125 mm is 75 mm**. A structure wider
than that is clipped, and clipping is silent — the mask channel simply stops at
the patch edge, and the model is trained on a truncated structure against a
full-size label.

`crop.py` measures every structure and reports the patch size the cohort needs:

```
structure size in plane: median 27 mm, max 87 mm
crop_size needed for train.py (at 0.78125 mm/px, gate 10 mm, augmentation):
   50th percentile:   80 px (62 mm FOV, covers 55%)
  100th percentile:  192 px (150 mm FOV, covers 100%)
  --> use --crop-size 192 to fit every case
```

`train.py` then refuses to start if any mask reaches the patch border, naming
the count and a size that would fit. `--allow-clipped` overrides it. The
required size accounts for the structure's own extent, the largest zoom-in, the
rotation swing, the gate band, and the translation — a 40 mm structure needs
112 px, not 52.

Two things to keep in mind when raising it. Compute grows quadratically, so 192
px is four times the cost of 96. And `crop.py --for-crop-size` must be given
the same value, or the disk crops will be too tight for the larger patch.

Raising `--in-plane-mm` instead widens the field of view at constant cost but
throws away detail, and it changes what the trained weights mean, so it is a
cohort-level decision rather than something to tune per run.

### Multi-label files

Label files may hold several structures as different integer values.
`--label-value N` selects one; it flows through cropping, training and
evaluation, and is stored in the checkpoint so `eval.py` uses the same one.

The selected label is binarized **before** any interpolation. Resampling a
multi-label image and thresholding afterwards is the obvious implementation and
it is wrong: halfway between label 2 and label 4 the interpolated value is 3, a
different structure entirely, so any neighbouring label that touches the target
gets pulled in — silently, because the result still looks like a plausible mask.

`crop.py` keeps every label inside the box by default, so one crop can serve
several targets; `--binarize-mask` writes the selected label alone as 0/1.

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

**Each plane is a 5 mm slab mean, whatever the native slice thickness.**
Sampling one plane every 5 mm from a 1 mm scan would use a fifth of the data
and give planes that are sharper and noisier than the same anatomy scanned at
5 mm, so a model trained on one would not transfer to the other. Instead each
plane averages `round(5 / spacing)` native-spaced samples around it — 5 at
1 mm, 2 at 2.5 mm — which is how a scanner builds a thick slice from thin
ones. At 5 mm or coarser it is the native slice itself, unchanged. On a
phantom scanned at both 1 mm and 5 mm the planes then agree to under 0.01 HU,
against 250 HU without averaging. Set it with `--slab-mm` (`0` takes a single
plane). The spacing stands in for the slice thickness, which NIfTI does not
record, so overlapping reconstructions are smoothed slightly more than
necessary. The mask channel is not averaged.

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

The 95% intervals `eval.py` writes are built in the standardized space and
mapped back (`TargetStandardizer.interval`). With `--log-target` they are
therefore asymmetric in the target's units, with the longer tail upwards,
rather than a symmetric mean ± 1.96σ from the delta method, which has the
wrong coverage and can reach far below zero. `z_score`, `z_std` and
`coverage_95` are computed in the same space, so they describe those
intervals.

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

136 tests covering resampling geometry against known slice positions, slab
averaging across slice thicknesses, log-space intervals, flipped
direction cosines, border clamping, falloff continuity, the gating intensity
domain, exact affine rotation on non-square images, the standardizer round
trip, the loss schedule, and an end-to-end run of all three CLIs against real
`.nii.gz` files with mismatched voxel and matrix sizes, including the
equivalence of the pipeline output before and after cropping and the isolation
of one label from a touching neighbour. Tests requiring
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
