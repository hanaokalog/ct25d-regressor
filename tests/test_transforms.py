"""Augmentation geometry and target standardization."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

from ct25d.transforms import (  # noqa: E402
    RandomAffine2D,
    TargetStandardizer,
    affine_theta,
    rescale_target,
)


def fixed(**kw):
    """A RandomAffine2D with no randomness left, for exact assertions."""
    base = dict(translate=0.0, scale=(1.0, 1.0), rotate_deg=0.0, shear=0.0,
                hflip=False, z_flip=False)
    base.update(kw)
    return RandomAffine2D(**base)


@pytest.fixture
def sample():
    x = torch.zeros(4, 64, 64)
    for c in range(3):
        x[c, 20:40, 24:44] = 100.0 * (c + 1)
    x[3, 28:36, 28:36] = 1.0
    return x


def test_identity_is_a_no_op(sample):
    out, scale = fixed()(sample)
    # grid_sample resamples in float32, so allow ~1e-6 relative error
    assert torch.allclose(out, sample, rtol=1e-5, atol=1e-2)
    assert float(scale) == pytest.approx(1.0)


def test_scaling_changes_the_mask_area_by_s_squared():
    # a large disc: with a small mask the area ratio is dominated by the
    # 0.5 threshold quantizing whole pixels
    H = W = 128
    yy, xx = np.mgrid[0:H, 0:W]
    x = torch.zeros(4, H, W)
    x[3] = torch.from_numpy(
        (((yy - 64) ** 2 + (xx - 64) ** 2) <= 30.0 ** 2).astype(np.float32))
    for s in (0.9, 1.1, 1.2):
        out, scale = fixed(scale=(s, s))(x)
        assert float(scale) == pytest.approx(s)
        ratio = float(out[3].sum()) / float(x[3].sum())
        assert ratio == pytest.approx(s ** 2, rel=0.02)


def test_translation_moves_the_content(sample):
    out, _ = fixed(translate=0.1)(sample)          # uniform(-0.1, 0.1)
    assert not torch.allclose(out, sample, atol=1e-3)
    assert float(out[3].sum()) > 0


def _apply(theta, img):
    grid = F.affine_grid(theta, (1, 1) + img.shape, align_corners=False)
    return F.grid_sample(img[None, None], grid, align_corners=False)[0, 0]


def test_rotation_by_90_degrees_is_exact():
    """rotate_deg is a range, so the matrix is built directly here."""
    img = torch.zeros(64, 64)
    img[10, 40] = 1.0
    out = _apply(affine_theta(90.0, height=64, width=64), img)
    # centre is (31.5, 31.5); (dx, dy) = (8.5, -21.5) rotated by +90
    # -> (21.5, 8.5) -> row 40, col 53
    assert np.unravel_index(int(out.argmax()), (64, 64)) == (40, 53)
    assert float(out.max()) == pytest.approx(1.0, abs=1e-4)


def test_rotation_stays_isotropic_on_non_square_images():
    """The reason the matrix is conjugated into pixel space before use."""
    img = torch.zeros(40, 60)
    img[19, 40] = 1.0                      # 10.5 px right of the centre column
    out = _apply(affine_theta(90.0, height=40, width=60), img)
    assert np.unravel_index(int(out.argmax()), (40, 60)) == (30, 30)


def test_translation_is_a_fraction_of_the_image_size():
    img = torch.zeros(40, 60)
    img[20, 30] = 1.0
    out = _apply(affine_theta(0.0, translate_x=0.1, translate_y=0.25,
                              height=40, width=60), img)
    assert np.unravel_index(int(out.argmax()), (40, 60)) == (30, 36)


def test_shear_preserves_area():
    img = torch.zeros(96, 96)
    img[36:60, 36:60] = 1.0
    out = _apply(affine_theta(0.0, shear_x=0.05, shear_y=-0.05,
                              height=96, width=96), img)
    # det([[1, shx], [shy, 1]]) = 1 - shx*shy = 1.0025
    assert float(out.sum()) / float(img.sum()) == pytest.approx(1.0025, rel=0.01)


def test_mask_stays_binary(sample):
    out, _ = RandomAffine2D(translate=0.1, scale=(0.9, 1.1), rotate_deg=5.0,
                            shear=0.03)(sample)
    assert set(torch.unique(out[3]).tolist()) <= {0.0, 1.0}


def test_all_channels_get_the_same_transform(sample):
    x = sample.clone()
    x[0] = x[3] * 500.0                       # image channel copies the mask
    out, _ = RandomAffine2D(translate=0.1, scale=(0.9, 1.1), rotate_deg=5.0,
                            shear=0.03, image_padding="zeros")(x)
    assert torch.allclose((out[0] > 250).float(), out[3], atol=1e-6)


def test_batched_input_gets_independent_transforms(sample):
    xb = sample.unsqueeze(0).repeat(16, 1, 1, 1)
    out, scale = RandomAffine2D(translate=0.1, scale=(0.9, 1.1))(xb)
    assert out.shape == xb.shape and scale.shape == (16,)
    assert float(scale.std()) > 0
    areas = out[:, 3].flatten(1).sum(1)
    assert float(areas.std()) > 0


def test_z_flip_reverses_the_slice_order_only(sample):
    torch.manual_seed(0)
    xb = sample.unsqueeze(0).repeat(64, 1, 1, 1)
    out, _ = fixed(z_flip=True)(xb)
    tol = dict(rtol=1e-5, atol=1e-2)
    rev = [bool(torch.allclose(out[i, 0], xb[i, 2], **tol)) for i in range(64)]
    assert 10 < sum(rev) < 54                    # roughly half, not all or none
    for i in range(64):
        if rev[i]:
            assert torch.allclose(out[i, 2], xb[i, 0], **tol)
        else:
            assert torch.allclose(out[i, 0], xb[i, 0], **tol)
        assert torch.allclose(out[i, 1], xb[i, 1], **tol)    # centre unmoved
        assert torch.allclose(out[i, 3], xb[i, 3], **tol)    # mask untouched


def test_border_padding_does_not_invent_air(sample):
    x = torch.full((4, 32, 32), -50.0)
    x[3] = 0.0
    x[3, 12:20, 12:20] = 1.0
    out, _ = fixed(scale=(0.5, 0.5), image_padding="border")(x)
    assert float(out[0].min()) == pytest.approx(-50.0, abs=1e-3)


# --------------------------------------------------------------------------- #
def test_rescale_target():
    assert rescale_target(10.0, 1.1, 0) == pytest.approx(10.0)
    assert rescale_target(10.0, 1.1, 1) == pytest.approx(11.0)
    assert rescale_target(10.0, 1.1, 2) == pytest.approx(12.1)


def test_standardizer_roundtrip():
    y = np.random.default_rng(0).normal(500, 120, 200)
    std = TargetStandardizer().fit(y)
    z = std.transform(y)
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-5
    assert np.allclose(std.inverse_transform(z), y, atol=1e-3)


def test_standardizer_log_roundtrip():
    y = np.random.default_rng(0).lognormal(5, 0.5, 200)
    std = TargetStandardizer(log_transform=True).fit(y)
    assert np.allclose(std.inverse_transform(std.transform(y)), y, rtol=1e-4)


def test_standardizer_converts_sigma_to_the_original_unit():
    y = np.random.default_rng(0).normal(500, 120, 500)
    std = TargetStandardizer().fit(y)
    mu, sigma = np.array([0.0]), np.array([1.0])
    m, s = std.inverse_transform(mu, sigma)
    assert m == pytest.approx(std.mean_) and s == pytest.approx(std.std_)


def test_standardizer_state_dict_roundtrip():
    std = TargetStandardizer(log_transform=True).fit([1.0, 2.0, 3.0])
    other = TargetStandardizer().load_state_dict(std.state_dict())
    v = np.array([2.0])
    assert other.transform(v) == pytest.approx(std.transform(v))


def test_unfitted_standardizer_raises():
    with pytest.raises(RuntimeError):
        TargetStandardizer().transform(np.array([1.0]))


def test_log_transform_rejects_negative_targets():
    with pytest.raises(ValueError):
        TargetStandardizer(log_transform=True).fit([-1.0, 2.0])
