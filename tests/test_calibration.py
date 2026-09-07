"""Uncertainty calibration diagnostics."""

import numpy as np
import pytest

from ct25d.calibration import fit_sigma_scale, uncertainty_report


def test_sigma_scale_recovers_a_known_factor():
    rng = np.random.default_rng(0)
    n = 20000
    mu = rng.normal(0, 1, n)
    true_sigma = rng.uniform(0.5, 2.0, n)
    y = mu + rng.normal(0, true_sigma)
    reported = true_sigma / 2.5             # systematically overconfident
    assert fit_sigma_scale(mu, reported, y) == pytest.approx(2.5, rel=0.03)


def test_report_on_well_calibrated_predictions():
    rng = np.random.default_rng(1)
    n = 20000
    mu = rng.normal(100, 20, n)
    sigma = rng.uniform(1.0, 5.0, n)
    y = mu + rng.normal(0, sigma)
    r = uncertainty_report(mu, sigma, y)
    assert r["z_std"] == pytest.approx(1.0, abs=0.03)
    assert r["coverage_95"] == pytest.approx(0.95, abs=0.01)
    assert r["sigma_scale"] == pytest.approx(1.0, abs=0.03)
    assert r["corr_abs_err"] > 0.3          # sigma tracks the actual error


def test_constant_sigma_is_flagged_as_uninformative():
    rng = np.random.default_rng(2)
    n = 5000
    mu = rng.normal(0, 1, n)
    sigma_true = rng.uniform(0.5, 3.0, n)
    y = mu + rng.normal(0, sigma_true)
    r = uncertainty_report(mu, np.full(n, sigma_true.mean()), y)
    assert abs(r["corr_abs_err"]) < 0.05    # calibrated on average, useless per case


def test_negative_sigma_is_rejected():
    with pytest.raises(ValueError):
        fit_sigma_scale([0.0], [-1.0], [1.0])
