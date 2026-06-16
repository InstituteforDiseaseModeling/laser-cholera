"""Tests for calc_log_likelihood_binomial() — Binomial-distribution log-likelihood helper."""

import contextlib
import logging

import numpy as np
import pytest
from scipy.special import gammaln

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_binomial


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


class TestCalcLogLikelihoodBinomial:
    """Tests for ``calc_log_likelihood_binomial``."""

    def test_errors_when_observed_and_estimated_lengths_differ(self):
        """Length mismatch raises.

        Failure implies the function silently proceeds on mismatched arrays.
        """
        with pytest.raises(ValueError, match="Lengths of observed and estimated must match"):
            calc_log_likelihood_binomial(
                observed=np.array([1, 2]),
                estimated=np.array([0.5]),
                trials=np.array([10, 10]),
                verbose=False,
            )

    def test_returns_nan_for_all_na_input(self):
        """All-NaN input returns NaN and logs the no-usable-data message.

        Failure implies the no-data path returns a numeric value or stays silent.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_binomial(
                observed=np.array([np.nan, np.nan]),
                estimated=np.array([np.nan, np.nan]),
                trials=np.array([np.nan, np.nan]),
                verbose=True,
            )
        assert np.isnan(ll)
        assert any("No usable data" in m for m in logs)

    def test_errors_on_negative_weights(self):
        """Negative weights raise.

        Failure implies the weighted-sum semantics are broken.
        """
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            calc_log_likelihood_binomial(
                observed=np.array([1, 2]),
                estimated=np.array([0.3, 0.5]),
                trials=np.array([10, 10]),
                weights=np.array([1, -1], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_sum_weights(self):
        """All-zero weights raise.

        Failure implies a divide-by-zero hides a configuration error.
        """
        with pytest.raises(ValueError, match="All weights are zero"):
            calc_log_likelihood_binomial(
                observed=np.array([1, 2]),
                estimated=np.array([0.3, 0.5]),
                trials=np.array([10, 10]),
                weights=np.array([0, 0], dtype=float),
                verbose=False,
            )

    def test_errors_on_observed_not_integer_or_out_of_range(self):
        """Non-integer or out-of-range observed counts raise.

        Failure implies invalid count data slips through to the binomial PMF.
        """
        # Non-integer
        with pytest.raises(ValueError, match="observed must be integer counts"):
            calc_log_likelihood_binomial(
                observed=np.array([1.5, 2.0]),
                estimated=np.array([0.3, 0.5]),
                trials=np.array([10, 10]),
                verbose=False,
            )
        # observed > trials
        with pytest.raises(ValueError, match="observed must be integer counts"):
            calc_log_likelihood_binomial(
                observed=np.array([11, 2]),
                estimated=np.array([0.3, 0.5]),
                trials=np.array([10, 10]),
                verbose=False,
            )

    def test_errors_on_non_positive_trials(self):
        """Zero or non-integer trial counts raise.

        Failure implies invalid binomial parameters reach the PMF.
        """
        with pytest.raises(ValueError, match="trials must be positive integers"):
            calc_log_likelihood_binomial(
                observed=np.array([0, 1]),
                estimated=np.array([0.3, 0.5]),
                trials=np.array([0, 10]),
                verbose=False,
            )

    def test_errors_on_estimated_outside_unit_interval(self):
        """Estimated probabilities <= 0 or >= 1 raise.

        Failure implies bad model output silently produces NaN.
        """
        with pytest.raises(ValueError, match=r"estimated probabilities must be in \(0, 1\)"):
            calc_log_likelihood_binomial(
                observed=np.array([1, 2]),
                estimated=np.array([0.3, 1.0]),
                trials=np.array([10, 10]),
                verbose=False,
            )

    def test_returns_finite_for_typical_input(self):
        """Typical count data returns a finite scalar.

        Failure implies the happy-path computation is broken.
        """
        ll = calc_log_likelihood_binomial(
            observed=np.array([3, 4, 2]),
            estimated=np.array([0.3, 0.5, 0.25]),
            trials=np.array([10, 10, 8]),
            verbose=False,
        )
        assert np.isfinite(ll)

    def test_matches_manual_log_likelihood_formula(self):
        """Unweighted output matches the manual binomial log-PMF formula.

        The PMF is ``C(n, k) * p^k * (1-p)^(n-k)``. Compared via lgamma form.
        Failure implies the binomial PMF parameterization is wrong.
        """
        obs = np.array([3, 4, 2], dtype=float)
        p = np.array([0.3, 0.5, 0.25])
        n = np.array([10, 10, 8], dtype=float)
        # log C(n,k) + k log p + (n-k) log (1-p)
        ll_vec = gammaln(n + 1) - gammaln(obs + 1) - gammaln(n - obs + 1) + obs * np.log(p) + (n - obs) * np.log(1 - p)
        manual = float(np.sum(ll_vec))

        result = calc_log_likelihood_binomial(
            observed=obs,
            estimated=p,
            trials=n,
            verbose=False,
        )
        assert result == pytest.approx(manual, rel=1e-9)

    def test_matches_manual_calculation_with_weights(self):
        """Weighted output matches the weighted manual formula.

        Failure implies weights are not applied element-wise.
        """
        obs = np.array([3, 4, 2], dtype=float)
        p = np.array([0.3, 0.5, 0.25])
        n = np.array([10, 10, 8], dtype=float)
        w = np.array([1.0, 2.0, 0.5])
        ll_vec = gammaln(n + 1) - gammaln(obs + 1) - gammaln(n - obs + 1) + obs * np.log(p) + (n - obs) * np.log(1 - p)
        manual = float(np.sum(w * ll_vec))

        result = calc_log_likelihood_binomial(
            observed=obs,
            estimated=p,
            trials=n,
            weights=w,
            verbose=False,
        )
        assert result == pytest.approx(manual, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
