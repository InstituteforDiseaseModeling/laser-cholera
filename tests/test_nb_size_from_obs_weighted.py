"""Tests for nb_size_from_obs_weighted() — internal helper for NB dispersion estimation.

Translated from test_nb_size_from_obs_weighted.R.

Note: R and Python use different random number generators, so `set.seed(42)` and
`random_state=42` produce different sequences. Tests are written against statistical
properties (k is finite, k >= k_min, etc.) rather than exact values, so the RNG
difference does not affect correctness.

R's `rnbinom(n, mu=mu, size=k)` maps to
`scipy.stats.nbinom.rvs(n=k, p=k/(k+mu), size=n, random_state=42)`.
Note: scipy requires integer n for nbinom; the k=0.5 case uses scipy directly since
numpy's `negative_binomial` requires a positive integer for its n parameter.
"""

import numpy as np
import scipy.stats

from laser.cholera.calc_model_likelihood import nb_size_from_obs_weighted


class TestNbSizeFromObsWeighted:
    """Tests for nb_size_from_obs_weighted, the weighted MoM NB dispersion estimator."""

    def test_poisson_data_var_lte_mean_returns_inf(self):
        """Constant data (variance = 0 <= mean) returns np.inf.

        Given 20 copies of the value 10 with uniform weights,
        when nb_size_from_obs_weighted is called,
        then k should be np.inf because variance does not exceed the mean.

        Failure implies the Poisson / sub-Poisson guard condition is broken and
        the function is returning a finite k where the data carries no NB signal.
        """
        # Constant data: var=0 <= mean=10
        x = np.full(20, 10.0)
        w = np.ones(20)
        k = nb_size_from_obs_weighted(x, w)
        assert np.isinf(k)

    def test_overdispersed_data_returns_finite_k(self):
        """Overdispersed NB data returns a finite k that respects the k_min floor.

        Given 100 draws from NB(mu=10, size=2) with uniform weights,
        when nb_size_from_obs_weighted is called,
        then k should be finite and at least k_min=3 (the default floor).

        Failure of the finite check implies the MoM estimator failed on clearly
        overdispersed data. Failure of the k >= 3 check implies k_min is not applied.
        """
        # R: set.seed(42); rnbinom(100, mu=10, size=2)
        x = scipy.stats.nbinom.rvs(n=2, p=2 / (2 + 10), size=100, random_state=42)
        w = np.ones(100)
        k = nb_size_from_obs_weighted(x, w)
        assert np.isfinite(k)
        assert k >= 3  # k_min floor

    def test_k_min_floor_is_applied(self):
        """When the MoM estimate falls below k_min, k_min is returned.

        Given 200 draws from NB(mu=10, size=0.5) — very overdispersed, true k ≈ 0.5
        — with uniform weights and k_min=3,
        when nb_size_from_obs_weighted is called,
        then k should be >= 3.

        Failure implies the k_min floor is not enforced, which would allow
        near-zero dispersion estimates that collapse the NB toward a point mass.
        """
        # R: set.seed(42); rnbinom(200, mu=10, size=0.5)
        # scipy handles non-integer n via the Gamma-Poisson mixture
        x = scipy.stats.nbinom.rvs(n=0.5, p=0.5 / (0.5 + 10), size=200, random_state=42)
        w = np.ones(200)
        k = nb_size_from_obs_weighted(x, w, k_min=3)
        assert k >= 3

    def test_k_max_cap_is_applied(self):
        """When the MoM estimate exceeds k_max, the result is capped at k_max or is Inf.

        Given 200 Poisson(100) draws — variance ≈ mean, so the NB estimate is very
        large — with k_min=1 and k_max=100,
        when nb_size_from_obs_weighted is called,
        then k should be <= 100 OR np.inf (Poisson regime where var <= mean).

        Failure implies the k_max cap is not enforced, allowing unbounded k values
        that effectively degrade the NB to Poisson without signalling it explicitly.
        """
        # R: set.seed(42); rpois(200, lambda=100)
        x = scipy.stats.poisson.rvs(100, size=200, random_state=42)
        w = np.ones(200)
        k = nb_size_from_obs_weighted(x, w, k_min=1, k_max=100)
        assert k <= 100 or np.isinf(k)  # Either capped or Inf (Poisson)

    def test_all_na_input_returns_inf(self):
        """All-NaN input returns np.inf.

        Given 10 NaN values with uniform weights,
        when nb_size_from_obs_weighted is called,
        then k should be np.inf because no finite observations exist.

        Failure implies the NaN-filtering logic is broken and the function attempts
        to compute on invalid data, likely producing NaN instead of Inf.
        """
        x = np.full(10, np.nan)
        w = np.ones(10)
        k = nb_size_from_obs_weighted(x, w)
        assert np.isinf(k)

    def test_zero_weight_entries_are_excluded(self):
        """Entries with zero weight are excluded from the dispersion estimate.

        Given 10 copies of 10 (constant, zero variance) followed by an outlier of
        1000, with the outlier's weight set to 0,
        when nb_size_from_obs_weighted is called,
        then k should be np.inf because the effective data (after excluding the
        zero-weight outlier) is constant with zero variance.

        Failure implies zero-weight entries are included in the computation, which
        would produce a finite k driven entirely by the excluded outlier.
        """
        x = np.array([10.0] * 10 + [1000.0])
        w = np.array([1.0] * 10 + [0.0])  # Zero-weight the outlier
        k = nb_size_from_obs_weighted(x, w)
        # Without outlier, data is constant -> Inf
        assert np.isinf(k)
