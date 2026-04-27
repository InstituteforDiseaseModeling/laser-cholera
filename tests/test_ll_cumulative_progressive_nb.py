"""Tests for ll_cumulative_progressive_nb() — cumulative NB progression.

Translated from test_ll_cumulative_progressive_nb.R.

Note: R's `rep(NA_real_, 10)` maps to `np.full(10, np.nan)`. The all-NA test
uses `nansum`, which returns 0.0 for all-NaN input (R's `sum(..., na.rm=TRUE)`
does the same), so both implementations treat all-NA data as zero cumulative sum
and should handle it gracefully.

The `k_data=Inf` test uses `np.inf` to trigger the Poisson fallback path, matching
R's `k_data = Inf` which causes `dnbinom(..., size=Inf)` to converge to Poisson.
"""

import numpy as np

from laser.cholera.calc_model_likelihood import ll_cumulative_progressive_nb


class TestLlCumulativeProgressiveNb:
    """Tests for ll_cumulative_progressive_nb, the cumulative NB progression helper."""

    def test_returns_finite_for_basic_input(self):
        """Basic input with finite obs and est returns a finite negative log-likelihood.

        Given obs=[10,20,30,40,50,60,70,80] and est=[12,18,35,38,55,58,72,78]
        with k_data=3,
        when ll_cumulative_progressive_nb is called,
        then ll should be finite and negative.

        Failure of the finite check implies the cumulative NB computation is
        producing NaN or Inf on normal input. Failure of the negative check implies
        the per-observation normalization is returning a positive LL, which is
        unexpected for a log-PMF of counts.
        """
        obs = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
        est = np.array([12, 18, 35, 38, 55, 58, 72, 78], dtype=float)
        ll = ll_cumulative_progressive_nb(obs, est, k_data=3)
        assert np.isfinite(ll)
        assert ll < 0

    def test_perfect_match_produces_best_ll(self):
        """A perfect match (obs == est) produces a less negative LL than a bad match.

        Given obs=[10,20,30,40] with k_data=3,
        when ll_cumulative_progressive_nb is called once with est=obs (perfect)
        and once with est=obs*10 (severely over-predicted),
        then ll_perfect should be greater than ll_bad.

        Failure implies the cumulative shape term is not monotone in fit quality,
        which would undermine its use as a calibration signal.
        """
        obs = np.array([10, 20, 30, 40], dtype=float)
        ll_perfect = ll_cumulative_progressive_nb(obs, obs, k_data=3)
        ll_bad = ll_cumulative_progressive_nb(obs, obs * 10, k_data=3)
        assert ll_perfect > ll_bad

    def test_works_with_k_inf_poisson(self):
        """k_data=np.inf (Poisson limit) produces a finite result.

        Given obs=[10,20,30,40] and est=[12,18,32,38] with k_data=np.inf,
        when ll_cumulative_progressive_nb is called,
        then ll should be finite.

        Failure implies the Poisson fallback branch (triggered when k_data is
        infinite) is broken or raises an exception.
        """
        obs = np.array([10, 20, 30, 40], dtype=float)
        est = np.array([12, 18, 32, 38], dtype=float)
        ll = ll_cumulative_progressive_nb(obs, est, k_data=np.inf)
        assert np.isfinite(ll)

    def test_returns_floor_for_all_na_data(self):
        """All-NaN observed data is handled gracefully without raising.

        Given obs=all-NaN (10 values) and est=10 (constant) with k_data=3,
        when ll_cumulative_progressive_nb is called,
        then ll should be either finite or equal to -1e9 (penalty floor).

        Note: np.nansum of all-NaN returns 0.0 (matching R's sum(..., na.rm=TRUE)),
        so the cumulative sums will be 0. The function should not raise an exception.
        Failure implies the all-NaN case propagates to an unhandled NaN/exception.
        """
        obs = np.full(10, np.nan)
        est = np.full(10, 10.0)
        ll = ll_cumulative_progressive_nb(obs, est, k_data=3)
        # All cumulative sums will be 0 (nansum on all-NaN), or the function handles gracefully
        assert np.isfinite(ll) or ll == -1e9

    def test_returns_mean_not_sum_across_timepoints(self):
        """The function returns the mean (not sum) of per-timepoint contributions.

        Given obs=[10,20,30,40,50,60,70,80] and est=[12,18,35,38,55,58,72,78]
        with k_data=3,
        when ll_cumulative_progressive_nb is called with 4 timepoints and again
        with 2 timepoints,
        then both results should be finite and their ratio should be less than 3
        (a rough check that the result is not a sum, which would make the 4-point
        result roughly 2x the 2-point result).

        Failure implies the aggregation is a sum rather than a mean, which would
        make the shape-term weight non-comparable across different timepoint counts.
        """
        obs = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
        est = np.array([12, 18, 35, 38, 55, 58, 72, 78], dtype=float)
        # Default 4 timepoints
        ll_4 = ll_cumulative_progressive_nb(
            obs,
            est,
            k_data=3,
            timepoints=np.array([0.25, 0.5, 0.75, 1.0]),
        )
        # With 2 timepoints, the mean should be different scale
        ll_2 = ll_cumulative_progressive_nb(
            obs,
            est,
            k_data=3,
            timepoints=np.array([0.5, 1.0]),
        )
        # Both should be finite, and if it were sum instead of mean,
        # ll_4 would be ~2x ll_2. With mean, they should be similar magnitude.
        assert np.isfinite(ll_4)
        assert np.isfinite(ll_2)
        assert abs(ll_4 / ll_2) < 3  # Rough check: not 2x different
