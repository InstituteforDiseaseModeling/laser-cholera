"""Tests for compute_wis_parametric_row() — WIS helper with 0.5 MAE coefficient.

Translated from test_compute_wis_parametric_row.R.

R's `rep(NA_real_, n)` maps to `np.full(n, np.nan)`.
R's `k_use = Inf` maps to `k_use=np.inf`.
R's `qnbinom(p, mu=mu, size=k)` maps to
`scipy.stats.nbinom.ppf(p, n=k, p=k/(k+mu))`.
R's `is.na(x)` maps to `np.isnan(x)`.
`expect_equal(a, b, tolerance=t)` maps to `assertAlmostEqual(a, b, delta=t)`.
"""

import numpy as np
import scipy.stats

from laser.cholera.calc_model_likelihood import compute_wis_parametric_row


class TestComputeWisParametricRow:
    """Tests for compute_wis_parametric_row, the WIS helper with 0.5 MAE coefficient."""

    def test_wis_returns_finite_for_simple_case(self):
        """WIS is finite and non-negative for a basic perfect-match input.

        Given y=est=[10,20,30] with uniform weights and default NB quantiles
        (k_use=3),
        when compute_wis_parametric_row is called,
        then wis should be finite and >= 0.

        Failure of the finite check implies the WIS computation produces NaN or
        Inf on normal input. Failure of the non-negative check implies a sign error
        in the interval score or MAE term.
        """
        y = np.array([10, 20, 30], dtype=float)
        est = np.array([10, 20, 30], dtype=float)
        w = np.array([1, 1, 1], dtype=float)
        probs = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
        wis = compute_wis_parametric_row(y, est, w, probs, k_use=3)
        assert np.isfinite(wis)
        assert wis >= 0  # WIS is non-negative

    def test_wis_returns_nan_for_all_na_input(self):
        """All-NaN observed input returns np.nan.

        Given y=all-NaN with est=10 (constant), uniform weights, and k_use=3,
        when compute_wis_parametric_row is called,
        then wis should be np.nan.

        Failure implies the function returns a numeric value for entirely missing
        observed data, which would be an uninformative score.
        """
        y = np.full(5, np.nan)
        est = np.full(5, 10.0)
        w = np.ones(5)
        probs = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
        wis = compute_wis_parametric_row(y, est, w, probs, k_use=3)
        assert np.isnan(wis)

    def test_wis_works_with_k_inf_poisson_quantiles(self):
        """k_use=np.inf (Poisson quantiles) produces a finite result.

        Given y=est=[10,20,30] with uniform weights and k_use=np.inf,
        when compute_wis_parametric_row is called,
        then wis should be finite.

        Failure implies the Poisson quantile branch (triggered when k_use is
        infinite) is broken or raises an exception.
        """
        y = np.array([10, 20, 30], dtype=float)
        est = np.array([10, 20, 30], dtype=float)
        w = np.array([1, 1, 1], dtype=float)
        probs = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
        wis = compute_wis_parametric_row(y, est, w, probs, k_use=np.inf)
        assert np.isfinite(wis)

    def test_wis_includes_0_5_mae_coefficient_bracher_2021(self):
        """WIS includes the 0.5 MAE coefficient from Bracher et al. (2021).

        With only the median in probs (no interval pairs), K=0 and denom=0.5,
        so WIS = (0.5 * MAE) / 0.5 = MAE.
        If the 0.5 factor were missing, the result would be (1.0 * MAE) / 0.5 = 2*MAE.

        Given y=[100], est=[50], w=[1], probs=[0.5], k_use=3,
        when compute_wis_parametric_row is called,
        then wis should equal abs(100 - qnbinom(0.5, mu=50, size=3)) within 0.01.

        Failure implies the MAE coefficient is wrong (likely 1.0 instead of 0.5),
        producing a WIS that is 2x the expected value.
        """
        y = np.array([100], dtype=float)
        est = np.array([50], dtype=float)
        w = np.array([1], dtype=float)
        probs = np.array([0.5])

        wis = compute_wis_parametric_row(y, est, w, probs, k_use=3)
        q_med = scipy.stats.nbinom.ppf(0.5, n=3, p=3 / (3 + 50))
        mae = abs(100 - q_med)
        # With 0.5 coefficient: WIS = (0.5 * mae) / 0.5 = mae
        assert abs(wis - mae) <= 0.01

    def test_wis_works_with_non_standard_quantiles(self):
        """WIS is finite for non-standard quantile levels.

        Given y=est=[10,20,30], uniform weights, and probs=[0.1,0.3,0.5,0.7,0.9]
        (non-standard, where complement matching via exact float comparison might
        fail),
        when compute_wis_parametric_row is called,
        then wis should be finite.

        Failure implies the quantile-pair matching logic breaks on quantile levels
        that are not the standard [0.025, 0.25, 0.5, 0.75, 0.975] set.
        """
        y = np.array([10, 20, 30], dtype=float)
        est = np.array([10, 20, 30], dtype=float)
        w = np.array([1, 1, 1], dtype=float)
        # Non-standard quantiles where (1-p) %in% uppers might fail with exact float comparison
        probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        wis = compute_wis_parametric_row(y, est, w, probs, k_use=3)
        assert np.isfinite(wis)
