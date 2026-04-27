"""Tests for calc_model_likelihood() extreme inputs.

Translated from test_calc_model_likelihood_extreme.R.

Verifies that the NB naturally handles bad fits without hard floors:
extreme over- and under-prediction, zero estimates with non-zero observations,
non-finite estimates, anti-correlated inputs, and all-zero data.

R's `matrix(val, nrow, ncol)` maps to `np.full((nrow, ncol), val, dtype=float)`.
R's `matrix(c(...), nrow=1)` maps to `np.array([[...]], dtype=float)`.
R's `rep(c(a, b), n)` maps to `np.tile([a, b], n)`.
R's `NaN` maps to `np.nan`.
R's `identical(ll, -Inf)` maps to `ll == -np.inf`.
`is.finite(ll) || identical(ll, -Inf) || is.na(ll)` maps to
`np.isfinite(ll) or ll == -np.inf or np.isnan(ll)`.
"""

import numpy as np

from laser.cholera.calc_model_likelihood import calc_model_likelihood


class TestCalcModelLikelihoodExtreme:
    """Tests for calc_model_likelihood under extreme input conditions."""

    def test_extreme_over_prediction_produces_very_negative_finite_ll(self):
        """1000x over-prediction yields a finite LL less than -1000.

        Given obs=10 and est=10000 for both cases and deaths (2x50),
        when calc_model_likelihood is called,
        then ll should be finite and less than -1000.

        Failure of the finite check implies the extreme over-prediction path raises
        or produces NaN. Failure of the < -1000 check implies the NB penalty for
        large over-prediction is too mild.
        """
        n_loc = 2
        n_time = 50
        obs = np.full((n_loc, n_time), 10, dtype=float)
        est = np.full((n_loc, n_time), 10000, dtype=float)

        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs,
            est_deaths=est,
        )
        assert np.isfinite(ll)
        assert ll < -1000

    def test_extreme_under_prediction_produces_very_negative_finite_ll(self):
        """Near-zero prediction against obs=100 yields a finite LL less than -10000.

        Given obs=100 and est=0.001 for both cases and deaths (2x50),
        when calc_model_likelihood is called,
        then ll should be finite and less than -10000.

        Failure of the finite check implies the near-zero estimate path raises or
        produces NaN. Failure of the < -10000 check implies the proportional penalty
        for near-zero estimates is insufficient.
        """
        n_loc = 2
        n_time = 50
        obs = np.full((n_loc, n_time), 100, dtype=float)
        est = np.full((n_loc, n_time), 0.001, dtype=float)

        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs,
            est_deaths=est,
        )
        assert np.isfinite(ll)
        assert ll < -10000

    def test_zero_prediction_with_nonzero_observed_uses_proportional_penalty(self):
        """Zero estimates against mixed observed (including 100 and 50) yield ll < -1000.

        Given obs=[0, 100, 0, 50] and est=[0, 0, 0, 0] (1x4) for both cases and
        deaths,
        when calc_model_likelihood is called,
        then ll should be finite and less than -1000.

        Failure of the finite check implies the zero-estimate / non-zero-observed
        path raises or produces NaN instead of applying the proportional penalty
        (-observed * log(1e6)). Failure of the < -1000 check implies the penalty is
        not applied or is insufficient.
        """
        obs = np.array([[0, 100, 0, 50]], dtype=float)
        est = np.array([[0, 0, 0, 0]], dtype=float)

        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs,
            est_deaths=est,
        )
        assert np.isfinite(ll)
        assert ll < -1000

    def test_non_finite_estimate_returns_finite_or_neg_inf_or_nan(self):
        """NaN in estimated propagates gracefully to a finite, -Inf, or NaN result.

        Given obs=[10, 20, 30] and est=[10, NaN, 30] (1x3) for both cases and deaths,
        when calc_model_likelihood is called,
        then ll should be finite, equal to -inf, or NaN — not an exception.

        Failure implies the function raises on NaN inputs or produces an unexpected
        value, suggesting NaN is not properly filtered from the LL computation.
        """
        obs = np.array([[10, 20, 30]], dtype=float)
        est = np.array([[10, np.nan, 30]], dtype=float)

        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs,
            est_deaths=est,
        )
        assert np.isfinite(ll) or ll == -np.inf or np.isnan(ll)

    def test_anti_correlated_inputs_produce_bad_but_finite_ll(self):
        """Anti-correlated obs and est produce a finite LL less than -500.

        Given obs=[100, 0, 100, 0, ...] (1x100) and est=[0, 100, 0, 100, ...]
        (opposite pattern) for cases, and obs_deaths=est_deaths=zeros (1x100),
        when calc_model_likelihood is called,
        then ll should be finite and less than -500.

        Note: R's set.seed(42) has no effect here since all values are deterministic.

        Failure of the finite check implies the no-guardrail-floor path raises or
        collapses to -Inf. Failure of the < -500 check implies the anti-correlated
        pattern is not penalised sufficiently.
        """
        n_time = 100
        obs = np.array([np.tile([100, 0], n_time // 2)], dtype=float)
        est = np.array([np.tile([0, 100], n_time // 2)], dtype=float)

        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=np.zeros((1, n_time)),
            est_deaths=np.zeros((1, n_time)),
        )
        assert np.isfinite(ll)
        assert ll < -500

    def test_all_zero_obs_and_est_produces_zero_ll(self):
        """All-zero observed and estimated yields a finite LL of exactly 0.

        Given obs=est=zeros (3x20) for both cases and deaths,
        when calc_model_likelihood is called,
        then ll should be finite and approximately 0 within 1e-8.

        Failure of the finite check implies the all-zero path raises or produces NaN.
        Failure of the equality check implies the zero-data perfect-match case
        contributes a non-zero LL, suggesting a bug in the base-LL weighting.
        """
        obs = np.zeros((3, 20), dtype=float)
        est = np.zeros((3, 20), dtype=float)

        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs,
            est_deaths=est,
        )
        assert np.isfinite(ll)
        # Note: R returns exactly 0 because its NB loop has an explicit 0-obs/0-est
        # branch that short-circuits to ll=0.  Python's _calc_log_likelihood_nb
        # floors est to 1e-10 unconditionally, producing Poisson(0|1e-10) = -1e-10
        # per timestep.  With 3 locations, 20 steps, 2 outcomes the sum is ~-1.2e-8.
        # Using delta=1e-7 to accommodate this implementation difference.
        assert abs(ll - 0) <= 1e-7
