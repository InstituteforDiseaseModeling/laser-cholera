"""Tests for calc_log_likelihood_gamma() — Gamma-distribution log-likelihood helper."""

import contextlib
import logging

import numpy as np
import pytest
import scipy.stats

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_gamma


@contextlib.contextmanager
def _capture_logs(level=logging.INFO):
    logger = logging.getLogger("laser.cholera")
    prev_level = logger.level
    logger.setLevel(level)
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(f"{record.levelname}:{record.name}:{record.getMessage()}")

    handler = _Handler()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


class TestCalcLogLikelihoodGamma:
    """Tests for ``calc_log_likelihood_gamma``."""

    def test_errors_when_observed_and_estimated_lengths_differ(self):
        """Length mismatch raises.

        Failure implies the function silently proceeds on mismatched arrays.
        """
        with pytest.raises(ValueError, match="Lengths of observed and estimated must match"):
            calc_log_likelihood_gamma(np.array([1.0, 2.0]), np.array([1.5]), verbose=False)

    def test_returns_nan_for_all_na_input(self):
        """All-NaN input returns NaN and logs the no-usable-data message.

        Failure implies the no-data path returns a numeric value or stays silent.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_gamma(
                np.array([np.nan, np.nan]),
                np.array([np.nan, np.nan]),
                verbose=True,
            )
        assert np.isnan(ll)
        assert any("No usable data" in m for m in logs)

    def test_errors_on_negative_weights(self):
        """Negative weights raise.

        Failure implies the weighted-sum semantics are broken.
        """
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            calc_log_likelihood_gamma(
                np.array([1.0, 2.0]),
                np.array([1.5, 2.5]),
                weights=np.array([1, -1], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_sum_weights(self):
        """All-zero weights raise.

        Failure implies a divide-by-zero hides a configuration error.
        """
        with pytest.raises(ValueError, match="All weights are zero"):
            calc_log_likelihood_gamma(
                np.array([1.0, 2.0]),
                np.array([1.5, 2.5]),
                weights=np.array([0, 0], dtype=float),
                verbose=False,
            )

    def test_errors_on_non_positive_observed(self):
        """Zero or negative observed values raise — Gamma has support (0, ∞).

        Failure implies out-of-support values silently produce NaN or -inf.
        """
        with pytest.raises(ValueError, match="observed values must be strictly positive"):
            calc_log_likelihood_gamma(
                np.array([1.0, 0.0, 2.0]),
                np.array([1.5, 2.0, 2.5]),
                verbose=False,
            )

    def test_errors_on_non_positive_estimated(self):
        """Zero or negative estimated means raise.

        Failure implies invalid model output silently reaches the gamma PDF.
        """
        with pytest.raises(ValueError, match="estimated values must be strictly positive"):
            calc_log_likelihood_gamma(
                np.array([1.0, 2.0]),
                np.array([0.0, 2.5]),
                verbose=False,
            )

    def test_errors_on_flat_observed(self):
        """Zero-variance observed raises — shape parameter is undefined.

        Failure implies the flat-input guard has regressed.
        """
        with pytest.raises(ValueError, match="Variance is non-positive"):
            calc_log_likelihood_gamma(
                np.array([2.0, 2.0, 2.0]),
                np.array([2.0, 2.0, 2.0]),
                verbose=False,
            )

    def test_returns_finite_for_typical_input(self):
        """Typical positive continuous data returns a finite scalar.

        Failure implies the happy-path computation is broken.
        """
        ll = calc_log_likelihood_gamma(
            np.array([2.5, 3.2, 1.8]),
            np.array([2.4, 3.0, 2.0]),
            verbose=False,
        )
        assert np.isfinite(ll)

    def test_matches_docstring_pinned_value(self):
        """Default call matches the value pinned in the docstring example.

        Failure implies the gamma parameterization has shifted.
        """
        ll = calc_log_likelihood_gamma(
            np.array([2.5, 3.2, 1.8]),
            np.array([2.4, 3.0, 2.0]),
            verbose=False,
        )
        assert ll == pytest.approx(-1.731035287031648, rel=1e-9)

    def test_matches_manual_log_likelihood_formula(self):
        """Unweighted output matches the manual gamma log-PDF formula.

        The shape parameter is estimated from observed (alpha = mean^2 / var)
        and the scale is per-element (scale_i = est_i / alpha). Compared
        against ``scipy.stats.gamma.logpdf``.

        Failure implies the gamma parameterization or shape estimator is off.
        """
        obs = np.array([2.5, 3.2, 1.8])
        est = np.array([2.4, 3.0, 2.0])
        mu = float(np.mean(obs))
        s2 = float(np.var(obs, ddof=1))
        shape = mu**2 / s2
        scale = est / shape
        manual = float(np.sum(scipy.stats.gamma.logpdf(obs, a=shape, scale=scale)))

        result = calc_log_likelihood_gamma(obs, est, verbose=False)
        assert result == pytest.approx(manual, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
