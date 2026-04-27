"""Reference numerical tests for calc_model_likelihood.

Translated from test_calc_model_likelihood_reference.R. These pin exact output
values for known inputs to verify cross-implementation equivalence. If these tests
break, the likelihood function's numerical behavior has changed.

Note: R's `calc_log_likelihood_negbin` and `calc_log_likelihood_poisson` (MOSAIC
public API) are accessed here via the private `_calc_log_likelihood_nb` helper,
which unifies both distributions. Pass `k=np.inf` for the Poisson case.

R's `expect_equal(..., tolerance=t)` uses relative tolerance. Tolerances here are
converted to absolute delta = tolerance * abs(expected_value), then rounded
conservatively. The extreme over-prediction test uses `delta=1` matching R's
`tolerance=1` verbatim.
"""

import unittest

import numpy as np

from laser.cholera.calc_model_likelihood import _calc_log_likelihood_nb
from laser.cholera.calc_model_likelihood import calc_model_likelihood

# Shared test data: 2 locations x 10 timesteps
# R:
#   ref_obs_c <- matrix(c(10,20,...,100, 5,10,...,50), nrow=2, byrow=TRUE)
#   ref_est_c <- matrix(c(12,18,...,105, 6, 9,...,52), nrow=2, byrow=TRUE)
#   ref_obs_d <- round(ref_obs_c * 0.05)
#   ref_est_d <- round(ref_est_c * 0.05)
REF_OBS_C = np.array(
    [
        [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
    ],
    dtype=float,
)

REF_EST_C = np.array(
    [
        [12, 18, 35, 38, 55, 58, 72, 78, 88, 105],
        [6, 9, 14, 22, 23, 32, 33, 42, 43, 52],
    ],
    dtype=float,
)

REF_OBS_D = np.round(REF_OBS_C * 0.05)
REF_EST_D = np.round(REF_EST_C * 0.05)


class TestCalcModelLikelihoodReference(unittest.TestCase):
    """Numerical regression tests that pin known outputs for known inputs.

    Each test corresponds directly to a `test_that` block in
    test_calc_model_likelihood_reference.R. Failure indicates that the likelihood
    function's numerical behavior has changed from the established reference.
    """

    def test_reference_core_nb_only_produces_known_value(self):
        """Core NB likelihood with no shape terms returns the pinned reference value.

        Given the 2x10 reference matrices and all shape-term weights at their default
        of 0,
        when calc_model_likelihood is called,
        then the result should equal -100.90235311 within tolerance 1e-4.

        Failure means the core NB log-likelihood or k-estimation logic has changed.
        """
        ll = calc_model_likelihood(REF_OBS_C, REF_EST_C, REF_OBS_D, REF_EST_D)
        self.assertAlmostEqual(ll, -100.90235311, delta=0.02)

    def test_reference_core_nb_plus_cumulative_produces_known_value(self):
        """Core NB + cumulative progression shape term returns the pinned reference value.

        Given the 2x10 reference matrices and weight_cumulative_total=0.25,
        when calc_model_likelihood is called,
        then the result should equal -102.4291 within tolerance 1e-4.

        Failure means the cumulative NB shape term or its T-normalization scale
        factor has changed.
        """
        ll = calc_model_likelihood(
            REF_OBS_C,
            REF_EST_C,
            REF_OBS_D,
            REF_EST_D,
            weight_cumulative_total=0.25,
        )
        self.assertAlmostEqual(ll, -102.4291, delta=0.02)

    def test_reference_core_nb_plus_wis_produces_known_value(self):
        """Core NB + WIS shape term returns the pinned reference value.

        Given the 2x10 reference matrices and weight_wis=0.10,
        when calc_model_likelihood is called,
        then the result should equal -102.8312 within tolerance 1e-4.

        Failure means the WIS computation or its T-normalization scale factor has
        changed.
        """
        ll = calc_model_likelihood(
            REF_OBS_C,
            REF_EST_C,
            REF_OBS_D,
            REF_EST_D,
            weight_wis=0.10,
        )
        self.assertAlmostEqual(ll, -102.8312, delta=0.02)

    def test_reference_perfect_match_produces_known_value(self):
        """Perfect match (obs == est) returns the pinned reference value.

        Given the 2x10 reference matrices passed as both observed and estimated,
        when calc_model_likelihood is called,
        then the result should equal -99.29492711 within tolerance 1e-4.

        Failure means the MoM k-estimation or NB log-PMF evaluation has changed.
        """
        ll = calc_model_likelihood(REF_OBS_C, REF_OBS_C, REF_OBS_D, REF_OBS_D)
        self.assertAlmostEqual(ll, -99.29492711, delta=0.02)

    def test_reference_extreme_1000x_over_prediction_produces_known_value(self):
        """Extreme 1000x over-prediction returns the pinned reference value.

        Given a 1x10 matrix with obs=10, est=10000 for cases and obs=1, est=1000
        for deaths,
        when calc_model_likelihood is called,
        then the result should equal -109160.93253574 within delta=1.

        The loose tolerance (delta=1) matches R's tolerance=1 verbatim and reflects
        that the exact value of very large negative scores is less important than
        their order of magnitude.
        """
        ll = calc_model_likelihood(
            np.full((1, 10), 10.0),
            np.full((1, 10), 10000.0),
            np.full((1, 10), 1.0),
            np.full((1, 10), 1000.0),
        )
        self.assertAlmostEqual(ll, -109160.93253574, delta=1)

    def test_reference_nb_element_level_ll_for_known_inputs(self):
        """NB element-level log-likelihood for known inputs returns the pinned reference value.

        Translates R's calc_log_likelihood_negbin (MOSAIC public function) via
        _calc_log_likelihood_nb. R uses weights=NULL (uniform), which maps to
        np.ones(n) here.

        Given observed=[10,20,30,40,50], estimated=[12,18,35,38,55], k=3,
        when _calc_log_likelihood_nb is called with uniform weights,
        then the result should equal -18.70319184 within tolerance 1e-4.

        Failure means the NB log-PMF evaluation or the scipy/R distribution
        parameterisation equivalence has changed.
        """
        observed = np.array([10, 20, 30, 40, 50], dtype=float)
        estimated = np.array([12, 18, 35, 38, 55], dtype=float)
        weights = np.ones(5)
        ll = _calc_log_likelihood_nb(observed, estimated, weights, k=3, k_min=3)
        self.assertAlmostEqual(ll, -18.70319184, delta=0.002)

    def test_reference_poisson_ll_for_perfect_match(self):
        """Poisson log-likelihood for a perfect match returns the pinned reference value.

        Translates R's calc_log_likelihood_poisson (MOSAIC public function) via
        _calc_log_likelihood_nb with k=np.inf (Poisson limit). R uses weights=NULL
        (uniform), which maps to np.ones(n) here.

        Given observed=estimated=[10,20,30] and k=inf (Poisson),
        when _calc_log_likelihood_nb is called with uniform weights,
        then the result should equal -7.12184753 within tolerance 1e-4.

        Failure means the Poisson log-PMF path in _calc_log_likelihood_nb has
        changed.
        """
        observed = np.array([10, 20, 30], dtype=float)
        estimated = np.array([10, 20, 30], dtype=float)
        weights = np.ones(3)
        ll = _calc_log_likelihood_nb(observed, estimated, weights, k=np.inf)
        self.assertAlmostEqual(ll, -7.12184753, delta=0.002)

    def test_reference_1x3_matrix_minimum_viable_input_produces_known_value(self):
        """Minimum viable 1x3 input returns the pinned reference value.

        Given 1x3 matrices with exactly 3 observations (the minimum required by
        the min_obs_for_likelihood threshold),
        when calc_model_likelihood is called,
        then the result should equal -15.49095 within tolerance 1e-3.

        Failure means boundary behavior at the minimum-observations threshold has
        changed.
        """
        ll = calc_model_likelihood(
            np.array([[50, 60, 70]], dtype=float),
            np.array([[55, 58, 72]], dtype=float),
            np.array([[5, 6, 7]], dtype=float),
            np.array([[4, 6, 8]], dtype=float),
        )
        self.assertAlmostEqual(ll, -15.49095, delta=0.02)

    def test_reference_1x1_matrix_returns_0_below_min_obs_threshold(self):
        """A 1x1 matrix (below the min_obs_for_likelihood threshold of 3) returns 0.

        Given 1x1 matrices with a single observation, which is below the minimum
        of 3 observations required for a meaningful NB likelihood,
        when calc_model_likelihood is called,
        then the result should be 0 because no location contributes any LL.

        Note: in the R reference this uses tolerance=1e-8 (essentially exact). The
        value 0 arises because have_cases and have_deaths are both False, so
        ll_cases and ll_deaths default to 0 and the location contributes 0.
        """
        ll = calc_model_likelihood(
            np.array([[50.0]]),
            np.array([[55.0]]),
            np.array([[5.0]]),
            np.array([[4.0]]),
        )
        self.assertAlmostEqual(ll, 0, delta=1e-8)

    def test_reference_perfect_match_always_better_than_imperfect(self):
        """Perfect match likelihood is strictly greater than imperfect match likelihood.

        Given two model runs — one where est==obs and one where est is the reference
        estimated matrices,
        when calc_model_likelihood is called for each,
        then the perfect-match score must be strictly greater than the imperfect score.

        Failure implies the likelihood function is no longer monotone in fit quality,
        which would break any optimizer or sampler relying on it.
        """
        ll_perfect = calc_model_likelihood(REF_OBS_C, REF_OBS_C, REF_OBS_D, REF_OBS_D)
        ll_close = calc_model_likelihood(REF_OBS_C, REF_EST_C, REF_OBS_D, REF_EST_D)
        self.assertGreater(ll_perfect, ll_close)


if __name__ == "__main__":
    unittest.main()
