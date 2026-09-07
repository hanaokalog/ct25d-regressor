"""Distance gating: physical distances, smooth edges, correct intensity domain."""

import numpy as np
import pytest

from ct25d.constants import CT_WINDOW, TARGET_INPLANE_MM
from ct25d.gating import (
    DistanceGate,
    falloff,
    gate_image,
    make_input,
    signed_distance_channel,
    window_01,
)

PX = TARGET_INPLANE_MM


@pytest.fixture
def disc():
    H = W = 129
    yy, xx = np.mgrid[0:H, 0:W]
    return ((yy - 64) ** 2 + (xx - 64) ** 2 <= 8.0 ** 2).astype(np.float32)


@pytest.mark.parametrize("profile", ["linear", "cosine", "smoothstep", "gaussian"])
def test_falloff_endpoints(profile):
    t = np.array([0.0, 1.0])
    w = falloff(t, profile)
    assert np.isclose(w[0], 1.0) and np.isclose(w[1], 0.0, atol=1e-9)


@pytest.mark.parametrize("profile", ["linear", "cosine", "smoothstep", "gaussian"])
def test_falloff_is_monotone(profile):
    w = falloff(np.linspace(0, 1, 101), profile)
    assert np.all(np.diff(w) <= 1e-12)


def test_cosine_is_flat_at_both_ends_but_linear_is_not():
    t = np.linspace(0, 1, 2001)
    for prof, limit in (("cosine", 0.02), ("smoothstep", 0.02)):
        d = np.abs(np.gradient(falloff(t, prof), t))
        assert d[0] < limit and d[-1] < limit
    d_lin = np.abs(np.gradient(falloff(t, "linear"), t))
    assert d_lin[0] > 0.9          # the kink the smooth profiles remove


def test_distance_is_in_millimetres(disc):
    g = DistanceGate(radius_mm=10.0, pixel_mm=PX)
    d = g.distance_mm(disc)
    assert d[disc > 0].max() == 0.0
    # a point 20 px to the right of a disc of radius 8 px
    assert np.isclose(d[64, 84], (20 - 8) * PX, atol=PX)


def test_weight_is_one_inside_and_zero_beyond_the_radius(disc):
    g = DistanceGate(radius_mm=10.0, pixel_mm=PX, profile="cosine")
    w = g.weight(disc, 3)
    assert w.shape == (3, 129, 129)
    assert np.allclose(w[:, disc > 0], 1.0)
    far = g.distance_mm(disc) >= 10.0
    assert np.allclose(w[0][far], 0.0)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)


def test_floor_keeps_a_trace_of_the_surroundings(disc):
    w = DistanceGate(10.0, PX, floor=0.1).weight(disc, 1)
    assert np.isclose(w.min(), 0.1)
    assert np.isclose(w.max(), 1.0)


def test_sphere_mode_attenuates_the_neighbouring_slices(disc):
    w = DistanceGate(10.0, PX, z_offsets_mm=(-5.0, 0.0, 5.0)).weight(disc, 3)
    assert np.isclose(w[1].max(), 1.0)
    assert np.isclose(w[0].max(), 0.5, atol=1e-6)
    assert (w[0] > 0).sum() < (w[1] > 0).sum()


def test_sphere_mode_checks_the_slice_count(disc):
    with pytest.raises(ValueError):
        DistanceGate(10.0, PX, z_offsets_mm=(-5.0, 0.0, 5.0)).weight(disc, 2)


def test_empty_mask_gives_zero_weight_everywhere(disc):
    w = DistanceGate(10.0, PX).weight(np.zeros_like(disc), 1)
    assert np.allclose(w, 0.0)


def test_gating_converges_to_the_window_floor_not_to_water(disc):
    """The reason gating happens in [0, 1] and not in HU."""
    hu = np.full((1,) + disc.shape, 300.0, np.float32)
    g = DistanceGate(10.0, PX)
    out = gate_image(hu, disc, g, out_range=(0.0, 1.0))
    far = g.weight(disc, 1)[0] == 0.0
    assert np.allclose(out[0][far], 0.0)                       # background
    lo, hi = CT_WINDOW
    assert np.isclose(out[0][far][0] * (hi - lo) + lo, lo)     # = -100 HU, not 0 HU


def test_window_01_clips_and_scales():
    x = np.array([-1024.0, -100.0, 450.0, 1000.0, 3000.0], np.float32)
    w = window_01(x)
    assert np.allclose(w, [0.0, 0.0, 0.5, 1.0, 1.0])


def test_signed_distance_channel(disc):
    sdf = signed_distance_channel(disc, PX, clip_mm=10.0)
    assert sdf[64, 64] < 0 and sdf[0, 0] == 1.0
    assert -1.0 <= sdf.min() and sdf.max() <= 1.0
    assert abs(sdf[64, 72]) < 0.2                 # near the boundary


def test_make_input_channel_layout(stack_hu):
    g = DistanceGate(10.0, PX)
    assert make_input(stack_hu, g, mask_channel="sdf").shape[0] == 4
    assert make_input(stack_hu, g, mask_channel="none").shape[0] == 3
    assert make_input(stack_hu, g, mask_channel="binary",
                      keep_context=True).shape[0] == 5
    with pytest.raises(ValueError):
        make_input(stack_hu, g, mask_channel="nope")


def test_make_input_context_channel_is_not_gated(stack_hu):
    out = make_input(stack_hu, DistanceGate(10.0, PX), keep_context=True)
    gated_centre, context = out[1], out[3]
    assert context.min() < gated_centre.min() + 1e-6
    assert (np.abs(context) > 1e-6).sum() > (np.abs(gated_centre + 1) > 1e-6).sum()


def test_make_input_without_gate_keeps_the_whole_image(stack_hu):
    out = make_input(stack_hu, gate=None, mask_channel="binary")
    assert out[:3].min() >= -1.0 and out[:3].max() <= 1.0
    assert np.unique(out[3]).tolist() == [0.0, 1.0]
