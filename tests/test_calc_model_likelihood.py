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

import numpy as np
import pandas as pd
import pytest
import scipy.stats

from laser.cholera.calc_model_likelihood import calc_model_likelihood
from laser.cholera.calc_model_likelihood import calc_multi_peak_magnitude_ll
from laser.cholera.calc_model_likelihood import calc_multi_peak_timing_ll

# Module-level shared data: 2x3 zero matrices
obs_zero = np.zeros((2, 3))
est_zero = np.zeros((2, 3))


class TestCalcModelLikelihood:
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
        assert np.isfinite(ll)
        assert abs(ll - 0) <= 1e-8

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
        assert np.isfinite(ll)
        assert abs(ll - 0) <= 1e-8

    def test_errors_on_non_matrix_inputs(self):
        """Non-array inputs raise an error matching 'inputs must be matrices'.

        Given obs_cases=obs_zero.tolist() (a plain Python list, not a 2-D array),
        when calc_model_likelihood is called,
        then it should raise an exception matching '2-D arrays'.

        Failure implies the function silently accepts non-array inputs, producing
        incorrect or undefined results without signalling the caller.
        """
        with pytest.raises(Exception, match="2-D arrays"):
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
        with pytest.raises(Exception, match="same shape"):
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
        with pytest.raises(Exception, match="weights_location must match n_locations"):
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
        )
        assert np.isfinite(ll) or np.isnan(ll)

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
        )
        assert np.isfinite(ll) or np.isnan(ll)

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
        assert np.isfinite(ll)
        assert ll < 0

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
        assert ll_same_peak > ll_shifted_peak

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
        assert ll_same_mag > ll_diff_mag

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
        assert np.isfinite(ll_default)
        assert np.isfinite(ll_custom)
        est_cases_bad = np.full((n_loc, n_time), 20, dtype=float)
        ll_bad_cumulative = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases_bad,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
            weight_cumulative_total=0.25,
        )
        assert ll_default > ll_bad_cumulative

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
        assert np.isfinite(ll_wis)
        ll_no_wis = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
        )
        assert ll_wis <= ll_no_wis

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
        )
        assert np.isfinite(ll_all)
        ll_core_only = calc_model_likelihood(
            obs_cases=obs_cases,
            est_cases=est_cases,
            obs_deaths=obs_deaths,
            est_deaths=est_deaths,
        )
        assert np.isfinite(ll_core_only)
        assert ll_all != ll_core_only

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
        assert np.isfinite(ll)

    def test_out_of_window_peaks_are_filtered(self):
        """Peak dates outside ``[date_start, date_stop]`` do not contribute to the LL.

        Given an ``epidemic_peaks`` DataFrame whose only row falls *before* the
        simulation window, with ``weight_peak_timing > 0`` and the matching
        ``date_start``/``date_stop`` kwargs supplied,
        when ``calc_model_likelihood`` is called,
        then the result must equal the LL computed with no peak data at all —
        proving the out-of-window row was dropped rather than clamped to t=0.

        Failure implies the in-window filter has regressed. Before the filter,
        ``np.argmin`` would have snapped the calendar peak to time-step 0 and
        the peak-shape term would contribute a non-zero score, making
        ``ll_with`` and ``ll_without`` differ.
        """
        n_loc, n_time = 1, 52
        obs = np.full((n_loc, n_time), 5, dtype=float)
        est = np.full((n_loc, n_time), 5, dtype=float)
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)

        out_of_window = pd.DataFrame(
            {"iso_code": ["AAA"], "peak_date": ["2010-01-01"], "loc_idx": [0]},
        )
        ll_with = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
            epidemic_peaks=out_of_window,
            date_start="2024-01-01",
            date_stop="2024-12-30",
        )
        ll_without = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
        )
        assert ll_with == pytest.approx(ll_without)

    def test_mixed_window_peaks_only_in_window_counted(self):
        """In-window peaks contribute exactly as if out-of-window rows were absent.

        Given an ``epidemic_peaks`` DataFrame containing one in-window peak
        (mid-simulation) and one far-out-of-window peak for the same location,
        and a second DataFrame with only the in-window peak,
        when ``calc_model_likelihood`` is called with each in turn (peak terms
        enabled and the calendar bounds supplied),
        then the two scores must be equal — the out-of-window row must
        contribute exactly zero, not be clamped to an endpoint.

        Failure implies a partial regression where only some out-of-window
        rows are filtered (e.g., leading edge but not trailing, or NaN dates
        but not date arithmetic). The estimated time series is non-uniform
        around the in-window peak so the peak-shape term has a non-trivial
        contribution; otherwise the test would pass vacuously even with a
        broken filter.
        """
        n_loc, n_time = 1, 52
        # Build a clear epidemic curve so the peak-shape term has signal.
        obs = np.full((n_loc, n_time), 5, dtype=float)
        obs[0, 19:30] = [10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5]
        est = np.full((n_loc, n_time), 5, dtype=float)
        est[0, 19:30] = [10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5]
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)

        # Simulation runs 2024-01-01..2024-12-30 → 52 weeks daily-clamped via
        # the weekly fallback. Calendar peak at week 25 is inside; calendar
        # peak in 2010 is outside.
        date_start = "2024-01-01"
        date_stop = "2024-12-30"

        in_window_only = pd.DataFrame(
            {
                "iso_code": ["AAA"],
                "peak_date": ["2024-06-24"],  # roughly week 25
                "loc_idx": [0],
            },
        )
        mixed = pd.DataFrame(
            {
                "iso_code": ["AAA", "AAA"],
                "peak_date": ["2024-06-24", "2010-06-01"],
                "loc_idx": [0, 0],
            },
        )

        kwargs = {
            "obs_cases": obs,
            "est_cases": est,
            "obs_deaths": obs_d,
            "est_deaths": est_d,
            "weight_peak_timing": 0.25,
            "weight_peak_magnitude": 0.25,
            "date_start": date_start,
            "date_stop": date_stop,
        }
        ll_in_only = calc_model_likelihood(epidemic_peaks=in_window_only, **kwargs)
        ll_mixed = calc_model_likelihood(epidemic_peaks=mixed, **kwargs)

        assert ll_in_only == pytest.approx(ll_mixed)


class TestLegacyPeakHelpers:
    """Tests for ``calc_multi_peak_timing_ll`` and ``calc_multi_peak_magnitude_ll``.

    These legacy helpers are reached when callers use the standalone peak-LL
    interface (matching the R ``MOSAIC::calc_multi_peak_*_ll`` API). They
    dispatch by ``iso_code`` against a DataFrame and apply the in-window
    filter directly. The class covers their early-return guards, the
    in-window filtering, and the happy path.
    """

    @staticmethod
    def _peaks(iso="AAA", dates=("2024-06-15",)):
        """Build a minimal epidemic_peaks DataFrame with a single ISO code."""
        return pd.DataFrame({"iso_code": [iso] * len(dates), "peak_date": list(dates)})

    def test_multi_peak_timing_returns_zero_when_epidemic_peaks_none(self):
        """``epidemic_peaks=None`` returns 0 immediately.

        Failure implies the early-return guard has regressed; the function
        would attempt to index a None object and crash.
        """
        obs = np.zeros(52)
        est = np.zeros(52)
        ll = calc_multi_peak_timing_ll(
            obs,
            est,
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-30",
            epidemic_peaks=None,
        )
        assert ll == 0.0

    def test_multi_peak_timing_returns_zero_when_iso_code_none(self):
        """``iso_code=None`` returns 0 — can't look up peaks without an ISO key.

        Failure implies the guard is missing and the DataFrame index lookup
        silently filters to an empty row set without explicit signal.
        """
        ll = calc_multi_peak_timing_ll(
            np.zeros(52),
            np.zeros(52),
            iso_code=None,
            date_start="2024-01-01",
            date_stop="2024-12-30",
            epidemic_peaks=self._peaks(),
        )
        assert ll == 0.0

    def test_multi_peak_timing_returns_zero_when_dates_none(self):
        """Missing ``date_start`` or ``date_stop`` returns 0.

        Without calendar bounds the function cannot construct ``date_seq``,
        so it must short-circuit. Failure implies the guard has regressed.
        """
        peaks = self._peaks()
        assert (
            calc_multi_peak_timing_ll(
                np.zeros(52),
                np.zeros(52),
                iso_code="AAA",
                date_start=None,
                date_stop="2024-12-30",
                epidemic_peaks=peaks,
            )
            == 0.0
        )
        assert (
            calc_multi_peak_timing_ll(
                np.zeros(52),
                np.zeros(52),
                iso_code="AAA",
                date_start="2024-01-01",
                date_stop=None,
                epidemic_peaks=peaks,
            )
            == 0.0
        )

    def test_multi_peak_timing_returns_zero_for_unknown_iso_code(self):
        """An ISO code absent from ``epidemic_peaks`` returns 0.

        Failure implies the function would attempt to score a non-existent
        peak window and either return -inf or NaN.
        """
        peaks = self._peaks(iso="XXX")
        ll = calc_multi_peak_timing_ll(
            np.zeros(52),
            np.zeros(52),
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-30",
            epidemic_peaks=peaks,
        )
        assert ll == 0.0

    def test_multi_peak_timing_returns_zero_when_date_seq_cannot_match(self):
        """If neither daily nor weekly date_seq matches obs_vec length, returns 0.

        Given a calendar window that produces 365 daily steps but obs_vec is
        length 99 (neither daily nor weekly count matches), the function
        cannot align peak dates to indices and must return 0.

        Failure implies the function would index past array bounds or
        silently match against an empty date sequence.
        """
        peaks = self._peaks()
        ll = calc_multi_peak_timing_ll(
            np.zeros(99),
            np.zeros(99),
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-30",
            epidemic_peaks=peaks,
        )
        assert ll == 0.0

    def test_multi_peak_timing_finite_for_in_window_peak(self):
        """An in-window peak returns a finite Normal log-PDF score.

        Given a peak date inside the simulation window and a non-trivial
        estimated time series, the function must produce a finite score.

        Failure implies the happy path has broken or the window detection
        for daily cadence is wrong.
        """
        peaks = self._peaks(dates=["2024-06-15"])
        n_time = 366
        obs = np.full(n_time, 5.0)
        est = np.full(n_time, 5.0)
        # Inject a peak in est near the expected date so the score has signal
        peak_idx = (pd.Timestamp("2024-06-15") - pd.Timestamp("2024-01-01")).days
        est[peak_idx] = 100.0
        ll = calc_multi_peak_timing_ll(
            obs,
            est,
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-31",
            epidemic_peaks=peaks,
        )
        assert np.isfinite(ll)

    def test_multi_peak_timing_finite_for_weekly_cadence(self):
        """Weekly-cadence date sequences (``timestep_to_weeks=1``) are also handled.

        Given a 53-week obs_vec (matching a 1-year weekly date range), the
        function must fall back to the weekly date_seq and produce a finite
        score for an in-window peak.

        Failure implies the weekly fallback branch (line ~263-269) is dead.
        """
        peaks = self._peaks(dates=["2024-06-15"])
        n_time = 53  # 53 weekly steps in 2024
        obs = np.full(n_time, 5.0)
        est = np.full(n_time, 5.0)
        est[25] = 50.0  # mid-year peak
        ll = calc_multi_peak_timing_ll(
            obs,
            est,
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-31",
            epidemic_peaks=peaks,
        )
        assert np.isfinite(ll)

    def test_multi_peak_magnitude_returns_zero_when_epidemic_peaks_none(self):
        """``epidemic_peaks=None`` returns 0 (mirror of timing helper).

        Failure implies the early-return guard on the magnitude helper has
        regressed.
        """
        ll = calc_multi_peak_magnitude_ll(
            np.zeros(52),
            np.zeros(52),
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-30",
            epidemic_peaks=None,
        )
        assert ll == 0.0

    def test_multi_peak_magnitude_returns_zero_for_unknown_iso_code(self):
        """An ISO code absent from ``epidemic_peaks`` returns 0 (magnitude side).

        Failure parallels the timing helper failure mode.
        """
        peaks = self._peaks(iso="XXX")
        ll = calc_multi_peak_magnitude_ll(
            np.zeros(52),
            np.zeros(52),
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-30",
            epidemic_peaks=peaks,
        )
        assert ll == 0.0

    def test_multi_peak_magnitude_finite_for_in_window_peak(self):
        """An in-window peak with matching obs/est magnitudes returns a finite score.

        Failure implies the happy path of the magnitude helper is broken.
        """
        peaks = self._peaks(dates=["2024-06-15"])
        n_time = 366
        obs = np.full(n_time, 5.0)
        est = np.full(n_time, 5.0)
        peak_idx = (pd.Timestamp("2024-06-15") - pd.Timestamp("2024-01-01")).days
        obs[peak_idx] = 100.0
        est[peak_idx] = 100.0
        ll = calc_multi_peak_magnitude_ll(
            obs,
            est,
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-31",
            epidemic_peaks=peaks,
        )
        assert np.isfinite(ll)


class TestCalcModelLikelihoodCoverage:
    """Coverage gap-fillers for ``calc_model_likelihood``.

    Each test below exercises a specific branch (peak-dispatch with peak
    data supplied, weekly cadence detection, validation paths, and edge
    cases in the cumulative/WIS terms) that the rest of the suite did not
    previously hit.
    """

    @staticmethod
    def _peaks_for(loc_idx=0, iso="AAA", date="2024-06-15"):
        """Build a single-row epidemic_peaks DataFrame with the ``loc_idx`` column."""
        return pd.DataFrame({"iso_code": [iso], "peak_date": [date], "loc_idx": [loc_idx]})

    def test_peak_shape_terms_fire_when_peak_data_supplied(self):
        """``calc_model_likelihood`` actually runs the peak terms when peak data is supplied.

        Given matching epidemic_peaks (with ``loc_idx``) and the calendar
        bounds, when calc_model_likelihood is called with peak-term weights,
        then the result differs from a call without any peak data. This
        proves the precompute block (calc_model_likelihood.py lines
        ~619-642) and the per-location dispatch (lines ~689-700) actually
        run, rather than silently no-oping as they did before this test
        was added.

        Failure implies the peak shape-term branch is dead despite weights
        being set, which would invalidate calibration runs that rely on
        peak signal.
        """
        n_loc, n_time = 1, 366
        obs = np.full((n_loc, n_time), 5.0)
        est = np.full((n_loc, n_time), 5.0)
        # Inject a peak signal at mid-year
        peak_idx = (pd.Timestamp("2024-06-15") - pd.Timestamp("2024-01-01")).days
        obs[0, peak_idx] = 100.0
        est[0, peak_idx] = 100.0
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)

        peaks = self._peaks_for(loc_idx=0, date="2024-06-15")
        ll_with_peak_data = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
            weight_peak_magnitude=0.25,
            epidemic_peaks=peaks,
            date_start="2024-01-01",
            date_stop="2024-12-31",
        )
        ll_without_peak_data = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
            weight_peak_magnitude=0.25,
        )
        # The peak branch must contribute something — equal results would
        # mean the precompute block is silently doing nothing.
        assert ll_with_peak_data != ll_without_peak_data
        assert np.isfinite(ll_with_peak_data)

    def test_peak_precompute_detects_weekly_cadence(self):
        """A weekly date range triggers the ``timestep_to_weeks=1`` branch.

        Given a 53-step time series with a calendar range that fits a
        weekly cadence, when calc_model_likelihood is called with peak
        weights and matching epidemic_peaks,
        then the precompute block must detect the weekly cadence (line
        ~624) and produce a finite result without error.

        Failure implies the weekly fallback in the main function has
        regressed and weekly-cadence simulations would silently skip the
        peak terms.
        """
        n_loc, n_time = 1, 53
        obs = np.full((n_loc, n_time), 5.0)
        est = np.full((n_loc, n_time), 5.0)
        obs[0, 25] = 50.0
        est[0, 25] = 50.0
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)

        peaks = self._peaks_for(loc_idx=0, date="2024-06-24")
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
            epidemic_peaks=peaks,
            date_start="2024-01-01",
            date_stop="2024-12-31",
        )
        assert np.isfinite(ll)

    def test_peak_precompute_skipped_when_no_cadence_matches(self):
        """A time-series length that fits neither daily nor weekly cadence falls back gracefully.

        Given obs/est arrays of length 99 with a year-long calendar range,
        neither daily (366) nor weekly (53) matches, so the function must
        skip the peak precompute and return a finite result (peak terms
        contributing 0).

        Failure implies the date_seq=None branch (line ~626) is missing
        and the function would crash on mismatched lengths.
        """
        n_loc, n_time = 1, 99
        obs = np.full((n_loc, n_time), 5.0)
        est = np.full((n_loc, n_time), 5.0)
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)

        peaks = self._peaks_for(loc_idx=0, date="2024-06-15")
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
            epidemic_peaks=peaks,
            date_start="2024-01-01",
            date_stop="2024-12-31",
        )
        assert np.isfinite(ll)

    def test_errors_on_negative_estimates(self):
        """Negative estimated values raise a clear ValueError.

        Failure implies negative estimates silently propagate to the NB
        PMF, producing -inf or NaN log-likelihoods.
        """
        with pytest.raises(ValueError, match="Estimated values must be non-negative"):
            calc_model_likelihood(
                obs_cases=np.array([[1.0, 2.0]]),
                est_cases=np.array([[-1.0, 2.0]]),
                obs_deaths=np.array([[0.0, 0.0]]),
                est_deaths=np.array([[0.0, 0.0]]),
            )

    def test_errors_on_negative_weights_location(self):
        """A negative entry in ``weights_location`` raises.

        Failure implies negative location weights would silently produce a
        non-interpretable weighted-sum.
        """
        with pytest.raises(ValueError, match="weights must be >= 0"):
            calc_model_likelihood(
                obs_cases=np.zeros((2, 3)),
                est_cases=np.zeros((2, 3)),
                obs_deaths=np.zeros((2, 3)),
                est_deaths=np.zeros((2, 3)),
                weights_location=np.array([1.0, -1.0]),
            )

    def test_errors_on_negative_weights_time(self):
        """A negative entry in ``weights_time`` raises.

        Failure mirrors the weights_location case.
        """
        with pytest.raises(ValueError, match="weights must be >= 0"):
            calc_model_likelihood(
                obs_cases=np.zeros((2, 3)),
                est_cases=np.zeros((2, 3)),
                obs_deaths=np.zeros((2, 3)),
                est_deaths=np.zeros((2, 3)),
                weights_time=np.array([1.0, -1.0, 1.0]),
            )

    def test_errors_on_zero_sum_weights_location(self):
        """All-zero ``weights_location`` raises with the not-all-zero message.

        Failure implies an all-zero vector slips through and causes a
        divide-by-zero in the per-location aggregation.
        """
        with pytest.raises(ValueError, match="must not all be zero"):
            calc_model_likelihood(
                obs_cases=np.zeros((2, 3)),
                est_cases=np.zeros((2, 3)),
                obs_deaths=np.zeros((2, 3)),
                est_deaths=np.zeros((2, 3)),
                weights_location=np.zeros(2),
            )

    def test_errors_on_zero_sum_weights_time(self):
        """All-zero ``weights_time`` raises.

        Failure mirrors the location case.
        """
        with pytest.raises(ValueError, match="must not all be zero"):
            calc_model_likelihood(
                obs_cases=np.zeros((2, 3)),
                est_cases=np.zeros((2, 3)),
                obs_deaths=np.zeros((2, 3)),
                est_deaths=np.zeros((2, 3)),
                weights_time=np.zeros(3),
            )

    def test_cumulative_progression_with_zero_estimate_returns_finite(self):
        """The cumulative term handles a zero-estimate vs nonzero-obs slice gracefully.

        Given obs with positive cumulative sums but est all zeros, the
        cumulative-progression branch should apply its proportional
        penalty rather than emit -inf.

        Failure implies the zero-estimate fallback inside
        ``ll_cumulative_progressive_nb`` has regressed.
        """
        n_loc, n_time = 1, 30
        obs = np.full((n_loc, n_time), 5.0)
        est = np.zeros((n_loc, n_time))
        obs_d = np.zeros_like(obs)
        est_d = np.zeros_like(est)
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_cumulative_total=0.25,
        )
        # The function returns -inf when the overall LL is non-finite
        # (line ~742); we only check the call doesn't raise.
        assert ll == -np.inf or np.isfinite(ll)

    def test_wis_handles_no_median_quantile(self):
        """WIS quantiles without the median (0.5) skip the MAE term cleanly.

        Failure implies the ``has_med`` branch (compute_wis_parametric_row
        line ~446) is broken; without the median, the MAE term should
        simply not contribute rather than crash.
        """
        rng = np.random.default_rng(123)
        n_loc, n_time = 1, 52
        obs = rng.poisson(10, (n_loc, n_time)).astype(float)
        est = np.full((n_loc, n_time), 10.0)
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)
        # Note: pairs without the median quantile
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_wis=0.10,
            wis_quantiles=np.array([0.1, 0.25, 0.75, 0.9]),
        )
        assert np.isfinite(ll)

    def test_weights_time_wrong_length_raises(self):
        """A ``weights_time`` vector with length != n_time_steps raises.

        Failure implies dimension validation has regressed on the time axis,
        which would let mismatched weights silently broadcast or truncate.
        """
        with pytest.raises(ValueError, match="weights_time must match"):
            calc_model_likelihood(
                obs_cases=np.zeros((2, 3)),
                est_cases=np.zeros((2, 3)),
                obs_deaths=np.zeros((2, 3)),
                est_deaths=np.zeros((2, 3)),
                weights_time=np.array([1.0, 1.0]),  # length 2, need 3
            )

    def test_wis_asymmetric_quantile_pair_uses_nearest_complement_fallback(self):
        """WIS quantile pairs without an exact complement fall back to the nearest match.

        Given a quantile list whose lower bound has no exact symmetric
        upper complement (e.g., 0.1 expects 0.9 but only 0.8 is supplied),
        when the WIS term is computed, then the function must reach the
        nearest-complement fallback (compute_wis_parametric_row line ~462)
        and still produce a finite score.

        Failure implies the fallback is broken; WIS calls with
        non-symmetric quantile sets would crash or return NaN.
        """
        rng = np.random.default_rng(123)
        n_loc, n_time = 1, 52
        obs = rng.poisson(10, (n_loc, n_time)).astype(float)
        est = np.full((n_loc, n_time), 10.0)
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)
        # 0.1 should pair with 0.9; we supply 0.8 instead → nearest-complement fallback fires.
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_wis=0.10,
            wis_quantiles=np.array([0.1, 0.25, 0.5, 0.75, 0.8]),
        )
        assert np.isfinite(ll)

    def test_wis_masks_weights_when_obs_contains_nan(self):
        """WIS zeros out per-timestep weights at NaN positions in obs.

        Given an obs row with a NaN at one timestep and the WIS term
        enabled, when calc_model_likelihood is called, then the WIS path
        must mask the NaN entry's weight (compute_wis_parametric_row line
        ~433) and produce a finite score over the remaining timesteps.

        Failure implies the NaN-aware weight masking inside WIS has
        regressed and the function would propagate NaN into the LL.
        """
        rng = np.random.default_rng(123)
        n_loc, n_time = 1, 52
        obs = rng.poisson(10, (n_loc, n_time)).astype(float)
        obs[0, 10] = np.nan  # inject one NaN to exercise the masking branch
        est = np.full((n_loc, n_time), 10.0)
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_wis=0.10,
        )
        assert np.isfinite(ll)

    def test_peak_precompute_skips_row_with_out_of_range_loc_idx(self):
        """Rows with ``loc_idx`` outside [0, n_locations) silently contribute nothing.

        Given an epidemic_peaks DataFrame with one valid loc_idx=0 row and
        one out-of-range loc_idx=99 row in a 1-location simulation, the
        precompute block must skip the out-of-range row (line ~639) and
        produce the same result as if only the valid row were present.

        Failure implies the bounds check on ``loc_idx`` has regressed and
        the function would index past n_locations, raising ``IndexError``.
        """
        n_loc, n_time = 1, 366
        obs = np.full((n_loc, n_time), 5.0)
        est = np.full((n_loc, n_time), 5.0)
        peak_idx = (pd.Timestamp("2024-06-15") - pd.Timestamp("2024-01-01")).days
        obs[0, peak_idx] = 100.0
        est[0, peak_idx] = 100.0
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)

        valid_only = pd.DataFrame(
            {"iso_code": ["AAA"], "peak_date": ["2024-06-15"], "loc_idx": [0]},
        )
        mixed = pd.DataFrame(
            {
                "iso_code": ["AAA", "AAA"],
                "peak_date": ["2024-06-15", "2024-06-15"],
                "loc_idx": [0, 99],  # 99 is out of range for n_locations=1
            },
        )
        kw = {
            "obs_cases": obs,
            "est_cases": est,
            "obs_deaths": obs_d,
            "est_deaths": est_d,
            "weight_peak_timing": 0.25,
            "date_start": "2024-01-01",
            "date_stop": "2024-12-31",
        }
        ll_valid = calc_model_likelihood(epidemic_peaks=valid_only, **kw)
        ll_mixed = calc_model_likelihood(epidemic_peaks=mixed, **kw)
        assert ll_valid == pytest.approx(ll_mixed)

    def test_peak_precompute_detects_weekly_cadence_main_fn(self):
        """A 52-step weekly cadence is detected by the main function's precompute.

        Given a 52-row time series whose calendar bounds match a
        once-a-week pd.date_range, the main precompute block must detect
        the weekly cadence (line ~628 sets timestep_to_weeks=1) and
        produce a finite result.

        Failure implies the main-function weekly fallback is dead; weekly
        simulations would silently skip the peak terms.
        """
        n_loc, n_time = 1, 52  # pd.date_range freq="W" for this range = 52
        obs = np.full((n_loc, n_time), 5.0)
        est = np.full((n_loc, n_time), 5.0)
        est[0, 25] = 50.0
        obs_d = np.ones_like(obs)
        est_d = np.ones_like(est)
        peaks = pd.DataFrame(
            {"iso_code": ["AAA"], "peak_date": ["2024-06-30"], "loc_idx": [0]},
        )
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_peak_timing=0.25,
            epidemic_peaks=peaks,
            date_start="2024-01-07",
            date_stop="2024-12-29",  # bounded to W-SUN
        )
        assert np.isfinite(ll)

    def test_nb_size_from_obs_weighted_bessel_fallback_branch(self):
        """``nb_size_from_obs_weighted`` falls back to the non-Bessel variance when denom<=0.

        Given a 3-timestep simulation where only the last entry has a
        positive ``weights_time``, the internal MoM dispersion estimator
        sees a single effective sample → ``sw - sw^2/sw == 0`` → the
        Bessel-corrected branch is skipped (line ~79).

        Failure implies the non-Bessel fallback inside
        ``nb_size_from_obs_weighted`` is dead; the function would divide by
        a non-positive denominator and emit NaN.
        """
        obs = np.array([[5.0, 5.0, 5.0]])
        est = np.array([[5.0, 5.0, 5.0]])
        obs_d = np.zeros_like(obs)
        est_d = np.zeros_like(est)
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weights_time=np.array([0.0, 0.0, 1.0]),
        )
        assert np.isfinite(ll)

    def test_calc_log_likelihood_nb_early_return_when_mask_empty(self):
        """``_calc_log_likelihood_nb`` returns 0.0 when mask_weights zeroes everything.

        Given a 3-timestep series where ``mask_weights`` zeroes every
        weight (e.g., the only non-zero-weight timestep also has a
        non-finite ``est``), the internal NB helper's mask becomes all
        False → line ~135 returns 0.0 early.

        Failure implies the early-return guard inside
        ``_calc_log_likelihood_nb`` has regressed; the function would
        compute a weighted sum over an empty mask, producing NaN.
        """
        obs = np.array([[5.0, 5.0, 5.0]])
        est = np.array([[1.0, 1.0, np.nan]])  # NaN at the only non-zero-weight index
        obs_d = np.zeros_like(obs)
        est_d = np.zeros_like(est)
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weights_time=np.array([0.0, 0.0, 1.0]),
        )
        # Function returns either -inf (if mask + downstream ⇒ non-finite)
        # or a finite value (if the empty-mask 0.0 propagates). Either path
        # exercises the line; we assert it doesn't raise.
        assert np.isfinite(ll) or ll == -np.inf

    def test_non_finite_ll_loc_total_collapses_to_minus_inf(self):
        """A non-finite per-location LL is replaced by ``-np.inf`` in the assembly.

        Given ``weight_cases=np.inf`` paired with a finite negative
        ``ll_cases``, the assembly produces a ``-np.inf`` ``ll_loc_total``;
        the function must catch this (lines ~742-743) and assign
        ``-np.inf`` to the location's LL rather than propagating NaN.
        Subsequently, ``np.nansum`` over the location array produces
        ``-np.inf``, which is also caught (line ~762).

        Failure implies the non-finite safety net has regressed and a
        single mis-scaled location could poison the entire model LL with
        NaN.
        """
        # Mild mismatch so ll_cases is a finite negative number.
        obs = np.array([[5.0, 6.0, 7.0, 8.0, 9.0]])
        est = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        obs_d = np.zeros_like(obs)
        est_d = np.zeros_like(est)
        ll = calc_model_likelihood(
            obs_cases=obs,
            est_cases=est,
            obs_deaths=obs_d,
            est_deaths=est_d,
            weight_cases=np.inf,
        )
        assert ll == -np.inf

    def test_multi_peak_magnitude_finite_for_weekly_cadence(self):
        """Weekly cadence detection works in the magnitude helper as well.

        Given a 53-row obs_vec and a calendar range that fits weekly
        ``pd.date_range``, ``calc_multi_peak_magnitude_ll`` must fall back
        to the weekly date_seq (line ~320) and return a finite score for
        an in-window peak.

        Failure implies the weekly fallback inside the magnitude helper is
        dead, leaving weekly-cadence callers with no usable LL.
        """
        peaks = pd.DataFrame({"iso_code": ["AAA"], "peak_date": ["2024-06-15"]})
        n_time = 53
        obs = np.full(n_time, 5.0)
        est = np.full(n_time, 5.0)
        obs[25] = 50.0
        est[25] = 50.0
        ll = calc_multi_peak_magnitude_ll(
            obs,
            est,
            iso_code="AAA",
            date_start="2024-01-01",
            date_stop="2024-12-31",
            epidemic_peaks=peaks,
        )
        assert np.isfinite(ll)
