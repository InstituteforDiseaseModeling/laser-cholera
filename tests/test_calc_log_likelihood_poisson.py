"""Tests for calc_log_likelihood_poisson() — Poisson-distribution log-likelihood helper."""

import contextlib
import logging

import numpy as np
import pytest
import scipy.stats

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_poisson


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


class TestCalcLogLikelihoodPoisson:
    """Tests for ``calc_log_likelihood_poisson``."""

    def test_errors_when_observed_and_estimated_lengths_differ(self):
        """Length mismatch raises.

        Failure implies the function silently proceeds on mismatched arrays.
        """
        with pytest.raises(ValueError, match="Lengths of observed and estimated must match"):
            calc_log_likelihood_poisson(np.array([1, 2]), np.array([1.5]), verbose=False)

    def test_returns_nan_for_all_na_input(self):
        """All-NaN input returns NaN and logs the no-usable-data message.

        Failure implies the no-data path returns a numeric value or stays silent.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_poisson(
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
            calc_log_likelihood_poisson(
                np.array([1, 2], dtype=float),
                np.array([1.5, 2.0]),
                weights=np.array([1, -1], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_sum_weights(self):
        """All-zero weights raise.

        Failure implies a divide-by-zero hides a configuration error.
        """
        with pytest.raises(ValueError, match="All weights are zero"):
            calc_log_likelihood_poisson(
                np.array([1, 2], dtype=float),
                np.array([1.5, 2.0]),
                weights=np.array([0, 0], dtype=float),
                verbose=False,
            )

    def test_returns_finite_for_typical_input(self):
        """Typical count data returns a finite scalar.

        Failure implies the happy-path computation is broken.
        """
        ll = calc_log_likelihood_poisson(
            np.array([2, 3, 4], dtype=float),
            np.array([2.2, 2.9, 4.1]),
            verbose=False,
        )
        assert np.isfinite(ll)

    def test_matches_docstring_pinned_value(self):
        """Default call matches the value pinned in the docstring example.

        Failure implies the Poisson parameterization has shifted.
        """
        ll = calc_log_likelihood_poisson(
            np.array([2, 3, 4], dtype=float),
            np.array([2.2, 2.9, 4.1]),
            verbose=False,
        )
        assert ll == pytest.approx(-4.447965653589073, rel=1e-9)

    def test_matches_manual_log_likelihood_formula(self):
        """Unweighted output matches the manual Poisson log-PMF formula.

        Compared against ``scipy.stats.poisson.logpmf``. Failure implies the
        Poisson PMF parameterization is wrong.
        """
        obs = np.array([2, 3, 4, 5], dtype=float)
        est = np.array([2.2, 2.9, 4.1, 5.0])
        manual = float(np.sum(scipy.stats.poisson.logpmf(obs.astype(int), mu=est)))
        result = calc_log_likelihood_poisson(obs, est, verbose=False)
        assert result == pytest.approx(manual, rel=1e-9)

    def test_zero_estimate_with_nonzero_obs_applies_proportional_penalty(self):
        """A zero estimate paired with a nonzero observation triggers the penalty path.

        The penalty is ``-obs * log(1e6)``, applied only to the offending
        entry. Note: with the default ``zero_buffer=True``, estimated values
        are floored to ``1e-10`` *before* the penalty mask is computed, so
        the penalty path is only reachable with ``zero_buffer=False``.

        Failure implies the penalty has regressed and a zero prediction
        would produce -inf log-likelihood, killing entire calibration runs.
        """
        # One pair with est=0 and obs=5 → penalty = -5 * log(1e6) ≈ -69.0775
        ll = calc_log_likelihood_poisson(
            np.array([5, 3], dtype=float),
            np.array([0.0, 3.0]),
            zero_buffer=False,
            verbose=False,
        )
        # Manually compute: penalty term for obs[0]=5 + Poisson logpmf for obs[1]=3, mu=3
        expected = -5 * np.log(1e6) + float(scipy.stats.poisson.logpmf(3, mu=3.0))
        assert ll == pytest.approx(expected, rel=1e-9)

    def test_zero_buffer_false_errors_on_non_integer_observed(self):
        """``zero_buffer=False`` enforces strict non-negative integer observed values.

        Failure implies the strict-mode guard has regressed and non-integer
        counts would be silently rounded.
        """
        with pytest.raises(ValueError, match="observed must contain non-negative integer counts"):
            calc_log_likelihood_poisson(
                np.array([1.5, 2.0]),
                np.array([1.5, 2.0]),
                zero_buffer=False,
                verbose=False,
            )

    def test_overdispersion_warning_fires_for_high_variance_data(self):
        """High variance-to-mean ratio logs a warning suggesting NegBin instead.

        Failure implies the diagnostic warning has been removed; users would
        lose the hint that their data is better modeled with NegBin.
        """
        # Variance / mean ratio of [1, 1, 1, 50] is ~16, well above 1.5.
        obs = np.array([1, 1, 1, 50], dtype=float)
        est = np.array([1.0, 1.0, 1.0, 1.0])
        with _capture_logs(level=logging.WARNING) as logs:
            calc_log_likelihood_poisson(obs, est, verbose=False)
        assert any("overdispersion" in m.lower() for m in logs)

    def test_matches_manual_calculation_with_weights(self):
        """Weighted output matches the weighted manual formula.

        Failure implies weights are not applied element-wise.
        """
        obs = np.array([2, 3, 4], dtype=float)
        est = np.array([2.2, 2.9, 4.1])
        w = np.array([1.0, 2.0, 0.5])
        manual = float(np.sum(w * scipy.stats.poisson.logpmf(obs.astype(int), mu=est)))
        result = calc_log_likelihood_poisson(obs, est, weights=w, verbose=False)
        assert result == pytest.approx(manual, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
