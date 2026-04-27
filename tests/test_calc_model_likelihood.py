"""Tests for calc_model_likelihood() — comprehensive tests for all terms.

Translated from test_calc_model_likelihood.R.

R's `matrix(val, nrow=r, ncol=c)` creates a constant matrix; maps to
`np.full((r, c), val, dtype=float)`.
R's `matrix(NA_real_, nrow=r, ncol=c)` maps to `np.full((r, c), np.nan)`.
R's `set.seed(123)` maps to `rng = np.random.default_rng(123)`.
R's `rpois(n, lambda=l)` maps to `rng.poisson(l, n)`.
R's `rnorm(n, 0, sd)` maps to `rng.normal(0, sd, n)`.
R's `dnorm(10:40, mean=25, sd=5)` maps to
`scipy.stats.norm.pdf(np.arange(10, 41), loc=25, scale=5)`.
R's `round(x)` maps to `np.round(x)`.
R's 1-based index `[i, a:b]` (inclusive) maps to Python 0-based `[i-1, a-1:b]`.
R's `rbind(a, b)` maps to `np.vstack([a, b])`.
R's `sample(-1:1, n, replace=TRUE)` maps to
`rng.choice(np.array([-1, 0, 1]), size=n, replace=True)`.
R's `sample(1:5, n, replace=TRUE)` maps to `rng.integers(1, 6, n)`.
`expect_true(is.finite(ll) || is.na(ll))` maps to
`self.assertTrue(np.isfinite(ll) or np.isnan(ll))`.

Note: R and Python use different RNGs, so seeded outputs will differ numerically.
Tests check structural properties (finite, ordering, inequality) rather than
exact values so the RNG difference does not affect correctness.
"""

import unittest

import numpy as np
import scipy.stats

from laser.cholera.calc_model_likelihood import calc_model_likelihood

# Module-level shared data: 2x3 zero matrices
obs_zero = np.zeros((2, 3))
est_zero = np.zeros((2, 3))


class TestCalcModelLikelihood(unittest.TestCase):
    """Tests for calc_model_likelihood, the full multi-term likelihood function."""

    def test_zero_data_returns_finite_ll(self):
        """Zero observed and estimated yields a finite log-likelihood of 0.

        Given obs_cases=est_cases=obs_deaths=est_deaths=zeros (2x3),
        when calc_model_likelihood is called with default parameters,
        then ll should be finite and approximately equal to 0 within 1e-8.

        Failure of the finite check implies the zero-data path raises or produces NaN.
        Failure of the equality check implies auxiliary terms or base LL is non-zero
        for a perfect zero-data match.
        """
        ll = calc_model_likelihood(
            obs_cases=obs_zero,
            est_cases=est_zero,
            obs_deaths=obs_zero,
            est_deaths=est_zero,
        )
        self.assertTrue(np.isfinite(ll))
        self.assertAlmostEqual(ll, 0, delta=1e-8)

    def test_weights_do_not_affect_zero_data_result(self):
        """Non-default weight_cases and weight_deaths still yield finite ll=0 for zero data.

        Given obs/est all zeros with weight_cases=2 and weight_deaths=3,
        when calc_model_likelihood is called,
        then ll should be finite and approximately 0 within 1e-8.

        Failure implies the weight scaling introduces a non-zero contribution when
        there is no count data, suggesting a bug in the base-LL weighting logic.
        """
        ll = calc_model_likelihood(
            obs_cases=obs_zero,
            est_cases=est_zero,
            obs_deaths=obs_zero,
            est_deaths=est_zero,
            weight_cases=2,
            weight_deaths=3,
        )
        self.assertTrue(np.isfinite(ll))
        self.assertAlmostEqual(ll, 0, delta=1e-8)

    def test_errors_on_non_matrix_inputs(self):
        """Non-array inputs raise an error matching 'inputs must be matrices'.

        Given obs_cases=obs_zero.tolist() (a plain Python list, not a 2-D array),
        when calc_model_likelihood is called,
        then it should raise an exception matching '2-D arrays'.

        Failure implies the function silently accepts non-array inputs, producing
        incorrect or undefined results without signalling the caller.
        """
        with self.assertRaisesRegex(Exception, "2-D arrays"):
            calc_model_likelihood(
                obs_cases=obs_zero.tolist(),
                est_cases=est_zero,
                obs_deaths=obs_zero,
                est_deaths=est_zero,
            )

    def test_errors_on_dimension_mismatch(self):
        """Mismatched matrix dimensions raise an error.

        Given est_cases with shape (1, 3) while obs_cases has shape (2, 3),
        when calc_model_likelihood is called,
        then it should raise an exception matching 'same shape'.

        Failure implies the function proceeds on mismatched arrays, producing results
        for the wrong number of locations without signalling the caller.
        """
        est_bad = np.zeros((1, 3))
        with self.assertRaisesRegex(Exception, "same shape"):
            calc_model_likelihood(
                obs_cases=obs_zero,
                est_cases=est_bad,
                obs_deaths=obs_zero,
                est_deaths=obs_zero,
            )

    def test_errors_on_wrong_weight_vector_length(self):
        """A weights_location vector with wrong length raises an error.

        Given obs/est all zeros (2 locations) and weights_location=[1] (length 1),
        when calc_model_likelihood is called,
        then it should raise an exception matching 'weights_location must match n_locations'.

        Failure implies the function silently pads or truncates the weight vector,
        producing results for the wrong number of locations.
        """
        with self.assertRaisesRegex(Exception, "weights_location must match n_locations"):
            calc_model_likelihood(
                obs_cases=obs_zero,
                est_cases=est_zero,
                obs_deaths=obs_zero,
                est_deaths=est_zero,
                weights_location=np.array([1]),
                weights_time=np.array([1]),
            )

    def test_all_na_data_returns_finite_or_na(self):
        """All-NaN data is handled gracefully without raising an exception.

        Given obs_cases=est_cases=obs_deaths=est_deaths=all-NaN (2x3) with verbose=True,
        when calc_model_likelihood is called,
        then ll should be either finite or NaN.

        Failure implies the all-NaN case propagates to an unhandled exception or
        produces a non-NaN infinite value.
        """
        obs_na = np.full((2, 3), np.nan)
        est_na = np.full((2, 3), np.nan)
        ll = calc_model_likelihood(
            obs_cases=obs_na,
            est_cases=est_na,
            obs_deaths=obs_na,
            est_deaths=est_na,
            verbose=True,
        )
        self.assertTrue(np.isfinite(ll) or np.isnan(ll))

    def test_all_na_observed_with_real_estimates_returns_finite_or_na(self):
        """All-NaN observed with real estimates is handled gracefully.

        Given obs_cases=obs_deaths=all-NaN (1x2) and est_cases=est_deaths=[[1.2, 3.4]],
        when calc_model_likelihood is called with verbose=True,
        then ll should be either finite or NaN.

        Failure implies the function raises an exception or produces a non-NaN infinite
        value when observed data is entirely missing.
        """
        obs_na = np.full((1, 2), np.nan)
        est_real = np.array([[1.2, 3.4]])
        ll = calc_model_likelihood(
            obs_cases=obs_na,
            est_cases=est_real,
            obs_deaths=obs_na,
            est_deaths=est_real,
            verbose=True,
        )
        self.assertTrue(np.isfinite(ll) or np.isnan(ll))

    def test_correct_ll_for_simple_nonzero_data_core_terms_only(self):
        """Core NB terms produce a finite negative LL for perfect-match count data.

        Given obs=est=[[1, 1, 1]] for both cases and deaths (1x3),
        when calc_model_likelihood is called with default parameters,
        then ll should be finite and negative.

        The core distribution falls back to Poisson for constant data (var <= mean).
        Failure of the finite check implies the core NB/Poisson evaluation fails on
        simple count data. Failure of the negative check implies a sign error in the LL.
        """
        obs = np.array([[1, 1, 1]], dtype=float)
        est = np.array([[1, 1, 1]], dtype=float)
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs,
            est_deaths=est,
        )
        self.assertTrue(np.isfinite(ll))
        self.assertLess(ll, 0)

    def test_peak_timing_term_works_correctly(self):
        """The peak timing term penalizes a shifted estimated peak vs a matched peak.

        Given obs_cases with a clear peak at row 0, cols 19-29 (0-based) — translated
        from R's [1, 20:30] (1-based) — and est_cases with the same peak, vs
        est_cases_shifted with the peak at cols 24-34 (shifted by 5),
        when calc_model_likelihood is called with weight_peak_timing=0.25,
        then ll_same_peak should be greater than ll_shifted_peak.

        Failure implies the peak timing term does not increase the LL for a
        better-matched peak, undermining its use as a calibration signal.
        """
        n_loc = 2
        n_time = 52
        obs_cases = np.full((n_loc, n_time), 5, dtype=float)
        # R: obs_cases[1, 20:30] (1-based, inclusive) = Python [0, 19:30]
        obs_cases[0, 19:30] = [10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5]
        est_cases = np.full((n_loc, n_time), 5, dtype=float)
        est_cases[0, 19:30] = [10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5]
        obs_deaths = np.ones((n_loc, n_time), dtype=float)
        est_deaths = np.ones((n_loc, n_time), dtype=float)
        ll_same_peak = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_peak_timing=0.25,
        )
        # R: est_cases_shifted[1, 25:35] (1-based, inclusive) = Python [0, 24:35]
        est_cases_shifted = np.full((n_loc, n_time), 5, dtype=float)
        est_cases_shifted[0, 24:35] = [10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5]
        ll_shifted_peak = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases_shifted,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_peak_timing=0.25,
        )
        self.assertGreater(ll_same_peak, ll_shifted_peak)

    def test_peak_magnitude_term_works_correctly(self):
        """The peak magnitude term penalizes a mismatched estimated peak magnitude.

        Given obs_cases with a single peak of 100 at row 0, col 24 (0-based) —
        translated from R's [1, 25] (1-based) — and est_cases with the same peak
        vs est_cases_diff with peak magnitude 50 (half),
        when calc_model_likelihood is called with weight_peak_magnitude=0.25,
        then ll_same_mag should be greater than ll_diff_mag.

        Failure implies the peak magnitude term does not penalize magnitude mismatch,
        undermining its calibration use.
        """
        n_loc = 2
        n_time = 52
        obs_cases = np.full((n_loc, n_time), 5, dtype=float)
        # R: obs_cases[1, 25] (1-based) = Python [0, 24]
        obs_cases[0, 24] = 100
        est_cases = np.full((n_loc, n_time), 5, dtype=float)
        est_cases[0, 24] = 100
        obs_deaths = np.ones((n_loc, n_time), dtype=float)
        est_deaths = np.ones((n_loc, n_time), dtype=float)
        ll_same_mag = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_peak_magnitude=0.25,
        )
        est_cases_diff = np.full((n_loc, n_time), 5, dtype=float)
        est_cases_diff[0, 24] = 50
        ll_diff_mag = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases_diff,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_peak_magnitude=0.25,
        )
        self.assertGreater(ll_same_mag, ll_diff_mag)

    def test_progressive_cumulative_total_term_works_correctly(self):
        """Cumulative total term is finite for default/custom timepoints and penalizes mismatch.

        Given obs_cases=est_cases=10 and obs_deaths=est_deaths=2 (all 2x52),
        when calc_model_likelihood is called with weight_cumulative_total=0.25 and
        cumulative_timepoints=[0.25, 0.5, 0.75, 1.0] (default) and [0.33, 0.67, 1.0]
        (custom), then both should be finite; also est_cases_bad=20 (double) should
        produce a worse (lower) ll than the matched case.

        Failure of the finite checks implies the cumulative term fails on constant input.
        Failure of the comparison implies the cumulative term does not penalize a
        systematically over-predicted series.
        """
        n_loc = 2
        n_time = 52
        obs_cases = np.full((n_loc, n_time), 10, dtype=float)
        est_cases = np.full((n_loc, n_time), 10, dtype=float)
        obs_deaths = np.full((n_loc, n_time), 2, dtype=float)
        est_deaths = np.full((n_loc, n_time), 2, dtype=float)
        ll_default = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_cumulative_total=0.25,
            cumulative_timepoints=np.array([0.25, 0.5, 0.75, 1.0]),
        )
        ll_custom = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_cumulative_total=0.25,
            cumulative_timepoints=np.array([0.33, 0.67, 1.0]),
        )
        self.assertTrue(np.isfinite(ll_default))
        self.assertTrue(np.isfinite(ll_custom))
        est_cases_bad = np.full((n_loc, n_time), 20, dtype=float)
        ll_bad_cumulative = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases_bad,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_cumulative_total=0.25,
        )
        self.assertGreater(ll_default, ll_bad_cumulative)

    def test_wis_term_penalizes_uncertainty_correctly(self):
        """The WIS term reduces the likelihood relative to the no-WIS baseline.

        Given Poisson-drawn obs_cases (lambda=10) and constant est_cases=10, and
        Poisson-drawn obs_deaths (lambda=2) and constant est_deaths=2 (all 2x52),
        when calc_model_likelihood is called with weight_wis=0.10 vs without,
        then ll_wis should be finite and <= ll_no_wis.

        Failure of the finite check implies the WIS term fails on Poisson count data.
        Failure of the comparison implies the WIS term is added (beneficial) rather than
        subtracted (penalty), reversing its intended effect on calibration.
        """
        rng = np.random.default_rng(123)
        n_loc = 2
        n_time = 52
        obs_cases = rng.poisson(10, (n_loc, n_time)).astype(float)
        est_cases = np.full((n_loc, n_time), 10, dtype=float)
        obs_deaths = rng.poisson(2, (n_loc, n_time)).astype(float)
        est_deaths = np.full((n_loc, n_time), 2, dtype=float)
        ll_wis = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_wis=0.10,
            wis_quantiles=np.array([0.025, 0.25, 0.5, 0.75, 0.975]),
        )
        self.assertTrue(np.isfinite(ll_wis))
        ll_no_wis = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
        )
        self.assertLessEqual(ll_wis, ll_no_wis)

    def test_all_terms_work_together_without_conflict(self):
        """All likelihood terms together produce a finite result different from core-only.

        Given obs_cases built from a Gaussian-shaped epidemic curve at timepoints
        9-39 (0-based) — translated from R's [i, 10:40] (1-based) — with peak at 25
        and sd=5, est_cases=obs_cases + small Gaussian noise clipped at 0,
        obs_deaths=round(obs_cases * 0.1), est_deaths=round(est_cases * 0.1),
        when calc_model_likelihood is called with all term weights enabled vs core-only,
        then both should be finite and their values should differ.

        Failure of finite checks implies one or more terms fail on realistic data.
        Failure of the inequality implies the auxiliary terms have no net effect,
        suggesting they all return zero or cancel out.
        """
        rng = np.random.default_rng(123)
        n_loc = 2
        n_time = 52
        obs_cases = np.full((n_loc, n_time), 5, dtype=float)
        # R: obs_cases[i, 10:40] (1-based, inclusive) = Python [i, 9:40]; dnorm on 10:40
        for i in range(n_loc):
            obs_cases[i, 9:40] = np.round(5 + 20 * scipy.stats.norm.pdf(np.arange(10, 41), loc=25, scale=5) * 100)
        est_cases = obs_cases + rng.normal(0, 2, (n_loc, n_time))
        est_cases[est_cases < 0] = 0
        obs_deaths = np.round(obs_cases * 0.1)
        est_deaths = np.round(est_cases * 0.1)
        ll_all = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_peak_timing=0.25,
            weight_peak_magnitude=0.25,
            weight_cumulative_total=0.25,
            weight_wis=0.10,
            verbose=False,
        )
        self.assertTrue(np.isfinite(ll_all))
        ll_core_only = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
        )
        self.assertTrue(np.isfinite(ll_core_only))
        self.assertNotEqual(ll_all, ll_core_only)

    def test_automatic_distribution_selection_works(self):
        """Automatic distribution selection (Poisson vs NB) produces a finite result.

        Given obs_cases row 0 (0-based) = 10 + small perturbations in {-1, 0, 1}
        (low variance, Poisson regime) and row 1 = Poisson(10) * Uniform{1,...,5}
        (high variance, NB regime), stacked via np.vstack with est_cases=10 constant,
        when calc_model_likelihood is called,
        then ll should be finite.

        Failure implies the auto-distribution path raises or produces NaN on either
        the Poisson or NB branch for those data characteristics.

        Note: R's rbind(obs_cases_low_var[1, ], obs_cases_high_var[2, ]) (1-based)
        maps to np.vstack([obs_cases_low_var[0, :], obs_cases_high_var[1, :]]).
        """
        rng = np.random.default_rng(123)
        n_loc = 2
        n_time = 52
        obs_cases_low_var = np.full((n_loc, n_time), 10, dtype=float)
        # R: obs_cases_low_var[1, ] (1-based row 1) = Python [0, :]
        obs_cases_low_var[0, :] += rng.choice(np.array([-1, 0, 1]), size=n_time, replace=True)
        obs_cases_low_var[obs_cases_low_var < 0] = 0
        obs_cases_high_var = np.full((n_loc, n_time), 10, dtype=float)
        # R: obs_cases_high_var[2, ] (1-based row 2) = Python [1, :]
        obs_cases_high_var[1, :] = rng.poisson(10, n_time) * rng.integers(1, 6, n_time)
        est_cases = np.full((n_loc, n_time), 10, dtype=float)
        obs_deaths = np.full((n_loc, n_time), 2, dtype=float)
        est_deaths = np.full((n_loc, n_time), 2, dtype=float)
        obs_combined = np.vstack([obs_cases_low_var[0, :], obs_cases_high_var[1, :]])
        ll = calc_model_likelihood(
            obs_cases=obs_combined,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
        )
        self.assertTrue(np.isfinite(ll))


if __name__ == "__main__":
    unittest.main()
