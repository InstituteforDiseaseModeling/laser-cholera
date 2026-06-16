"""Tests for calc_log_likelihood_normal() — Normal-distribution log-likelihood helper."""

import contextlib
import logging

import numpy as np
import pytest
import scipy.stats

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_normal


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


class TestCalcLogLikelihoodNormal:
    """Tests for ``calc_log_likelihood_normal``."""

    def test_errors_when_observed_and_estimated_lengths_differ(self):
        """Length mismatch raises.

        Failure implies the function silently proceeds on mismatched arrays.
        """
        with pytest.raises(ValueError, match="Lengths of observed and estimated must match"):
            calc_log_likelihood_normal(np.array([1.0, 2.0, 3.0]), np.array([1.5]), verbose=False)

    def test_returns_nan_for_all_na_input(self):
        """All-NaN input returns NaN and logs the no-usable-data message.

        Failure implies the no-data path returns a numeric value or stays silent.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_normal(
                np.array([np.nan, np.nan, np.nan]),
                np.array([np.nan, np.nan, np.nan]),
                verbose=True,
            )
        assert np.isnan(ll)
        assert any("No usable data" in m for m in logs)

    def test_errors_on_fewer_than_three_observations(self):
        """Fewer than 3 non-missing observations raise.

        Normal LL requires enough points to estimate residual SD; the function
        enforces n >= 3 explicitly. Failure implies the n-guard has regressed.
        """
        with pytest.raises(ValueError, match="At least 3"):
            calc_log_likelihood_normal(np.array([1.0, 2.0]), np.array([1.5, 2.0]), verbose=False)

    def test_errors_on_negative_weights(self):
        """Negative weights raise.

        Failure implies the weighted-sum semantics are broken.
        """
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            calc_log_likelihood_normal(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.5, 2.0, 2.5]),
                weights=np.array([1, -1, 1], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_sum_weights(self):
        """All-zero weights raise.

        Failure implies a divide-by-zero hides a configuration error.
        """
        with pytest.raises(ValueError, match="All weights are zero"):
            calc_log_likelihood_normal(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.5, 2.0, 2.5]),
                weights=np.array([0, 0, 0], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_residual_sd(self):
        """Perfect-fit data (residual SD = 0) raises.

        Failure implies the degenerate case slips through to ``norm.logpdf``
        with scale=0, producing -inf or NaN.
        """
        with pytest.raises(ValueError, match="Standard deviation of residuals is non-positive"):
            calc_log_likelihood_normal(
                np.array([1.0, 2.0, 3.0]),
                np.array([1.0, 2.0, 3.0]),
                verbose=False,
            )

    def test_returns_finite_for_typical_input(self):
        """Typical continuous data returns a finite scalar.

        Failure implies the happy-path computation is broken.
        """
        ll = calc_log_likelihood_normal(
            np.array([1.2, 2.8, 3.1]),
            np.array([1.0, 3.0, 3.2]),
            verbose=False,
        )
        assert np.isfinite(ll)

    def test_matches_manual_log_likelihood_formula(self):
        """Unweighted output matches the manual normal log-PDF formula.

        Sigma is estimated from residuals via ``np.std(..., ddof=1)`` and the
        per-element log-PDF uses ``estimated`` as the mean. Compared against
        ``scipy.stats.norm.logpdf``.

        Failure implies the sigma estimator or the per-element parameterization is wrong.
        """
        obs = np.array([1.2, 2.8, 3.1, 4.0, 5.5])
        est = np.array([1.0, 3.0, 3.2, 4.1, 5.4])
        residuals = obs - est
        sigma = float(np.std(residuals, ddof=1))
        manual = float(np.sum(scipy.stats.norm.logpdf(obs, loc=est, scale=sigma)))

        result = calc_log_likelihood_normal(obs, est, verbose=False)
        assert result == pytest.approx(manual, rel=1e-9)

    def test_matches_manual_calculation_with_weights(self):
        """Weighted output matches the weighted manual formula.

        Failure implies weights are not applied element-wise.
        """
        obs = np.array([1.2, 2.8, 3.1, 4.0])
        est = np.array([1.0, 3.0, 3.2, 4.1])
        w = np.array([1.0, 2.0, 0.5, 1.5])
        residuals = obs - est
        sigma = float(np.std(residuals, ddof=1))
        manual = float(np.sum(w * scipy.stats.norm.logpdf(obs, loc=est, scale=sigma)))

        result = calc_log_likelihood_normal(obs, est, weights=w, verbose=False)
        assert result == pytest.approx(manual, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
