"""Tests for calc_log_likelihood_negbin() — NB log-likelihood helper.

Translated from test_calc_log_likelihood_negbin.R.

R's `expect_error(..., "msg")` maps to `assertRaisesRegex(Exception, "msg")`.
R's `expect_message(expr, "msg")` maps to `assertLogs` checking that the given
string appears in at least one emitted log record.
R's `is.na(x)` maps to `np.isnan(x)`.
R's `lgamma(x)` maps to `scipy.special.gammaln(x)`.
`weights=NULL` in R maps to `weights=None` in Python.
`k_min=0` disables the k_min floor; the Python function must accept this parameter.

Note: Tests 9 and 10 compute the NB log-likelihood manually using the standard
formula and compare against the function output with `k_min=0` to bypass the
dispersion floor.
"""

import contextlib
import logging

import numpy as np
import pytest
from scipy.special import gammaln

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_negbin
from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_poisson


@contextlib.contextmanager
def _capture_logs(level=logging.INFO):
    root = logging.getLogger()
    prev_level = root.level
    root.setLevel(level)
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(f"{record.levelname}:{record.name}:{record.getMessage()}")

    handler = _Handler()
    root.addHandler(handler)
    try:
        yield records
    finally:
        root.removeHandler(handler)
        root.setLevel(prev_level)


class TestCalcLogLikelihoodNegbin:
    """Tests for calc_log_likelihood_negbin, the public NB log-likelihood function."""

    def test_errors_when_observed_and_estimated_lengths_differ(self):
        """A length mismatch between observed and estimated raises an error.

        Given observed has 2 elements and estimated has 1,
        when calc_log_likelihood_negbin is called,
        then it should raise an exception matching "Lengths of observed and estimated
        must match".

        Failure implies the function silently proceeds on mismatched arrays, which
        would produce incorrect results without any signal to the caller.
        """
        with pytest.raises(Exception, match="Lengths of observed and estimated must match"):
            calc_log_likelihood_negbin(
                observed=np.array([1, 2], dtype=float),
                estimated=np.array([1], dtype=float),
                k=1,
                weights=None,
                verbose=False,
            )

    def test_returns_nan_for_all_na_input_with_message(self):
        """All-NaN input returns NaN and logs a 'No usable data' message.

        Given observed=NaN and estimated=NaN for both elements, with verbose=True,
        when calc_log_likelihood_negbin is called,
        then it should return NaN and emit a log message containing 'No usable data'.

        Failure of the NaN check implies the function returns a numeric value for
        entirely missing data. Failure of the message check implies the verbose path
        is silent, making it hard to diagnose bad inputs.
        """
        with _capture_logs() as log_ctx:
            ll = calc_log_likelihood_negbin(
                observed=np.array([np.nan, np.nan]),
                estimated=np.array([np.nan, np.nan]),
                k=1,
                weights=None,
                verbose=True,
            )
        assert np.isnan(ll)
        assert any("No usable data" in m for m in log_ctx)

    def test_errors_on_negative_weights(self):
        """Negative weights raise an error.

        Given weights=[1, -1],
        when calc_log_likelihood_negbin is called,
        then it should raise an exception matching "All weights must be >= 0".

        Failure implies negative weights are silently accepted, which would produce
        a log-likelihood that is not interpretable as a weighted sum.
        """
        with pytest.raises(Exception, match="All weights must be >= 0"):
            calc_log_likelihood_negbin(
                observed=np.array([1, 2], dtype=float),
                estimated=np.array([1, 2], dtype=float),
                k=1,
                weights=np.array([1, -1], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_sum_weights(self):
        """All-zero weights raise an error.

        Given weights=[0, 0],
        when calc_log_likelihood_negbin is called,
        then it should raise an exception matching "All weights are zero".

        Failure implies the function divides by a zero weight sum, producing NaN
        or Inf without an informative error.
        """
        with pytest.raises(Exception, match="All weights are zero"):
            calc_log_likelihood_negbin(
                observed=np.array([1, 2], dtype=float),
                estimated=np.array([1, 2], dtype=float),
                k=1,
                weights=np.array([0, 0], dtype=float),
                verbose=False,
            )

    def test_non_integer_observed_values_are_rounded_silently(self):
        """Non-integer observed values are rounded internally without error.

        Given observed=[1.5, 2.0] (rounds to [2, 2]) and observed with a
        float-precision near-integer (2.0000000000000004), no error should be raised
        and the result should be finite.

        Given observed=[-1, 2] (negative after rounding), an error matching
        "observed must contain non-negative integer counts" should be raised.

        Failure of the finite checks implies the rounding guard is absent.
        Failure of the error check implies negative observations are silently accepted.
        """
        # c(1.5, 2) rounds to c(2, 2) — no error
        ll = calc_log_likelihood_negbin(
            observed=np.array([1.5, 2], dtype=float),
            estimated=np.array([1, 2], dtype=float),
            k=1,
            weights=None,
            verbose=False,
        )
        assert np.isfinite(ll)

        # Float near-integers from parquet transport should not error
        ll2 = calc_log_likelihood_negbin(
            observed=np.array([2.0000000000000004, 3.0]),
            estimated=np.array([2, 3], dtype=float),
            k=3,
            weights=None,
            verbose=False,
        )
        assert np.isfinite(ll2)

        # Negative values still error (round doesn't help)
        with pytest.raises(Exception, match="observed must contain non-negative integer counts"):
            calc_log_likelihood_negbin(
                observed=np.array([-1, 2], dtype=float),
                estimated=np.array([1, 2], dtype=float),
                k=1,
                weights=None,
                verbose=False,
            )

    def test_cushions_zero_or_negative_estimates(self):
        """Zero or negative estimated values are cushioned to produce a finite result.

        Given observed=[1, 2] and estimated=[0, -5],
        when calc_log_likelihood_negbin is called,
        then ll should be finite.

        Failure implies the function passes zero/negative mu to the NB log-PMF,
        producing -Inf or NaN rather than a large finite penalty.
        """
        obs = np.array([1, 2], dtype=float)
        est = np.array([0, -5], dtype=float)
        ll = calc_log_likelihood_negbin(obs, est, k=1, verbose=False)
        assert np.isfinite(ll)

    def test_defaults_to_poisson_when_variance_lte_mean(self):
        """k=None with var <= mean logs a 'using Poisson' message and matches Poisson LL.

        Given observed=rep(1, 5) (var=0, mean=1) and k=None with verbose=True,
        when calc_log_likelihood_negbin is called,
        then it should log a message containing 'using Poisson' and the result
        should equal calc_log_likelihood_poisson on the same data within 1e-8.

        Failure of the message check implies the auto-estimation path is silent.
        Failure of the equality check implies the Poisson fallback is not correctly
        applied when variance does not exceed the mean.
        """
        obs = np.ones(5)
        est = obs.copy()
        with _capture_logs() as log_ctx:
            ll_nb = calc_log_likelihood_negbin(obs, est, k=None, verbose=True)
        assert any("using Poisson" in m for m in log_ctx)
        ll_pois = calc_log_likelihood_poisson(obs, est, verbose=False)
        assert abs(ll_nb - ll_pois) <= 1e-8

    def test_uses_k_min_floor_when_provided_k_is_too_small(self):
        """k below k_min is silently floored to k_min with a log message.

        Given observed=[0, 1, 2], estimated=[1, 1, 1], k=1, verbose=True,
        and default k_min=3,
        when calc_log_likelihood_negbin is called,
        then it should log a message containing 'k_min' and return a finite result.

        Failure of the message check implies the k floor is silent and undetectable.
        Failure of the finite check implies flooring k has broken the computation.
        """
        obs = np.array([0, 1, 2], dtype=float)
        est = np.array([1, 1, 1], dtype=float)
        # k=1 < k_min=3 (default), so function floors to k_min and messages about it
        with _capture_logs() as log_ctx:
            ll = calc_log_likelihood_negbin(obs, est, k=1, verbose=True)
        assert any("k_min" in m for m in log_ctx)
        assert np.isfinite(ll)

    def test_matches_manual_negbin_log_likelihood_formula(self):
        """Unweighted output matches the manual NB log-likelihood formula.

        Given observed=[0,1,2], estimated=[1,2,3], k=2, k_min=0,
        when calc_log_likelihood_negbin is called,
        then the result should equal the manually computed sum of NB log-PMF
        values within 1e-8.

        k_min=0 disables the dispersion floor so k=2 is used as-is.
        The manual formula is:
            lgamma(x+k) - lgamma(k) - lgamma(x+1)
            + k*log(k/(k+mu)) + x*log(mu/(k+mu))

        Failure implies the NB log-PMF evaluation or parameterisation is wrong.
        """
        observed = np.array([0, 1, 2], dtype=float)
        estimated = np.array([1, 2, 3], dtype=float)
        k_param = 2.0
        # Manual NB log-likelihood
        ll_vec = (
            gammaln(observed + k_param)
            - gammaln(k_param)
            - gammaln(observed + 1)
            + k_param * np.log(k_param / (k_param + estimated))
            + observed * np.log(estimated / (k_param + estimated))
        )
        ll_manual = float(np.sum(ll_vec))

        ll_func = calc_log_likelihood_negbin(
            observed=observed,
            estimated=estimated,
            k=k_param,
            k_min=0,  # Disable flooring so k=2 is used as-is
            weights=None,
            verbose=False,
        )
        assert abs(ll_func - ll_manual) <= 1e-8

    def test_matches_manual_calculation_with_weights(self):
        """Weighted output matches the manual weighted NB log-likelihood formula.

        Given observed=[0,1,2], estimated=[1,2,3], weights=[1,2,0.5], k=2, k_min=0,
        when calc_log_likelihood_negbin is called,
        then the result should equal sum(weights * ll_vec) within 1e-8.

        k_min=0 disables the dispersion floor so k=2 is used as-is.

        Failure implies weights are not applied correctly to the per-element
        log-PMF values.
        """
        observed = np.array([0, 1, 2], dtype=float)
        estimated = np.array([1, 2, 3], dtype=float)
        weights = np.array([1, 2, 0.5])
        k_param = 2.0
        ll_vec = (
            gammaln(observed + k_param)
            - gammaln(k_param)
            - gammaln(observed + 1)
            + k_param * np.log(k_param / (k_param + estimated))
            + observed * np.log(estimated / (k_param + estimated))
        )
        ll_manual = float(np.sum(weights * ll_vec))

        ll_func = calc_log_likelihood_negbin(
            observed=observed,
            estimated=estimated,
            k=k_param,
            k_min=0,  # Disable flooring so k=2 is used as-is
            weights=weights,
            verbose=False,
        )
        assert abs(ll_func - ll_manual) <= 1e-8
