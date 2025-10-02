import re
import unittest
import warnings

import numpy as np
import pytest
from scipy.special import gammaln

from laser_cholera.likelihood import calc_log_likelihood_beta
from laser_cholera.likelihood import calc_log_likelihood_binomial
from laser_cholera.likelihood import calc_log_likelihood_gamma
from laser_cholera.likelihood import calc_log_likelihood_negbin
from laser_cholera.likelihood import calc_log_likelihood_normal
from laser_cholera.likelihood import calc_log_likelihood_poisson
from laser_cholera.likelihood import get_model_likelihood
from laser_cholera.metapop.model import run_model


class LikelihoodTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the model with default parameters
        cls.model = run_model(None)
        cls.obs_cases = cls.model.params.reported_cases
        cls.sim_cases = cls.model.patches.incidence[1:].T  # ignore t=0 (initial conditions)
        cls.obs_deaths = cls.model.params.reported_deaths
        cls.sim_deaths = cls.model.patches.disease_deaths[1:].T  # ignore t=0 (initial conditions)
        cls.n_locations, cls.n_steps = cls.obs_cases.shape

        cls.n_locations, cls.n_steps = cls.obs_cases.shape

    def test_get_model_likelihood(self):
        # test cases - poisson, deaths - poisson
        # test cases - poisson, deaths - normal (gaussian)
        # test cases - normal (gaussian), deaths - poisson
        # test cases - normal (gaussian), deaths - normal (gaussian)

        return

    def test_get_model_likelihood_reported_all_nan(self):
        nans = np.full_like(self.obs_cases, np.nan, dtype=np.float64)
        ll = get_model_likelihood(obs_cases=nans, sim_cases=self.sim_cases, obs_deaths=nans, sim_deaths=self.sim_deaths)
        assert np.all(np.isnan(ll))

    def test_get_model_likelihood_invalid_inputs(self):
        # obs_cases isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=[list(row) for row in self.obs_cases],
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
            )

        # sim_cases isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=[list(row) for row in self.sim_cases],
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
            )

        # obs_deaths isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=[list(row) for row in self.obs_deaths],
                sim_deaths=self.sim_deaths,
            )

        # sim_deaths isn't a NumPy array
        with pytest.raises(TypeError, match=r"obs_\* and est_\* must be numpy arrays"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=[list(row) for row in self.sim_deaths],
            )

        # obs_cases and sim_cases have different shapes
        with pytest.raises(ValueError, match=r"obs_\* and est_\* must have the same dimensions \(n_locations x n_time_steps\)\."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases[:, :-1],  # lop off last timestep
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
            )

        # obs_cases and obs_deaths have different shapes
        with pytest.raises(ValueError, match=r"obs_\* and est_\* must have the same dimensions \(n_locations x n_time_steps\)\."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths[:, :-1],  # lop off last timestep
                sim_deaths=self.sim_deaths,
            )

        # obs_cases and sim_deaths have different shapes
        with pytest.raises(ValueError, match=r"obs_\* and est_\* must have the same dimensions \(n_locations x n_time_steps\)\."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths[:, :-1],  # lop off last timestep
            )

        # len(weights_location) != self.n_locations
        with pytest.raises(ValueError, match="weights_location must have length"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weights_location=np.ones(self.n_locations - 1),
            )

        # len(weights_time) != self.n_steps
        with pytest.raises(ValueError, match="weights_time must have length"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weights_time=np.ones(self.n_steps - 1),
            )

        # some weights_location are negative
        with pytest.raises(ValueError, match="weights_location must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weights_location=np.random.random(self.n_locations) - 1,
            )

        # some weights_time are negative
        with pytest.raises(ValueError, match="weights_time must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weights_time=np.random.random(self.n_steps) - 1,
            )

        # all weights_location are zero
        with pytest.raises(ValueError, match=r"weights_location and weights_time must not all be zero."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weights_location=np.zeros(self.n_locations),
            )

        # all weights_time are zero
        with pytest.raises(ValueError, match=r"weights_location and weights_time must not all be zero."):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weights_time=np.zeros(self.n_steps),
            )

        # weight_cases < 0
        with pytest.raises(ValueError, match="weight_cases must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weight_cases=-0.8,
            )

        # weight_deaths < 0
        with pytest.raises(ValueError, match="weight_deaths must be >= 0"):
            get_model_likelihood(
                obs_cases=self.obs_cases,
                sim_cases=self.sim_cases,
                obs_deaths=self.obs_deaths,
                sim_deaths=self.sim_deaths,
                weight_deaths=-0.8,
            )

        return

    def test_calc_log_likelihood(self):
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23]
        # _beta = calc_log_likelihood_beta(obs_cases, sim_cases, verbose=False)
        # _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), verbose=False)
        # _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, verbose=False)
        # _normal = calc_log_likelihood_normal(obs_cases, sim_cases, verbose=False)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, verbose=False)

        return

    def test_calc_log_likelihood_no_weights(self):
        return

    def test_calc_log_likelihood_nan_observed(self):
        obs_cases = np.full_like(self.obs_cases[23], np.nan, dtype=np.float64)
        sim_cases = self.sim_cases[23]
        _beta = calc_log_likelihood_beta(obs_cases, sim_cases, verbose=False)
        _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), verbose=False)
        _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, verbose=False)
        _normal = calc_log_likelihood_normal(obs_cases, sim_cases, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, verbose=False)

        return

    def test_calc_log_likelihood_nan_simulated(self):
        obs_cases = self.obs_cases[23]
        sim_cases = np.full_like(self.sim_cases[23], np.nan, dtype=np.float64)
        _beta = calc_log_likelihood_beta(obs_cases, sim_cases, verbose=False)
        _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), verbose=False)
        _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, verbose=False)
        _normal = calc_log_likelihood_normal(obs_cases, sim_cases, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, verbose=False)

        return

    def test_calc_log_likelihood_nan_weights(self):
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23]
        weights = np.full_like(obs_cases, np.nan, dtype=np.float64)

        _beta = calc_log_likelihood_beta(obs_cases, sim_cases, weights=weights, verbose=False)
        _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, weights=weights, verbose=False)
        _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, weights=weights, verbose=False)
        _normal = calc_log_likelihood_normal(obs_cases, sim_cases, weights=weights, verbose=False)
        _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_nan_trials(self):
        # Just binomial
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23]
        _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.full_like(obs_cases, np.nan, dtype=np.float32), verbose=False)

        return

    def test_calc_log_likelihood_length_mismatches(self):
        # len(simulated) != len(observed)
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23][:-1]  # lop off last timestep

        with pytest.raises(ValueError, match="Lengths of observed "):
            _beta = calc_log_likelihood_beta(obs_cases, sim_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _normal = calc_log_likelihood_normal(obs_cases, sim_cases, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, verbose=False)

        # len(weights) != len(observed)
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23]
        weights = np.full_like(obs_cases[:-1], np.nan, dtype=np.float64)  # lop off last timestep

        with pytest.raises(ValueError, match="Lengths of observed "):
            _beta = calc_log_likelihood_beta(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _normal = calc_log_likelihood_normal(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="Lengths of observed "):
            _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_negative_weights(self):
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23]
        weights = -np.random.rand(len(obs_cases))

        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _beta = calc_log_likelihood_beta(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _normal = calc_log_likelihood_normal(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_zero_weights(self):
        obs_cases = self.obs_cases[23]
        sim_cases = self.sim_cases[23]
        weights = np.zeros_like(obs_cases)

        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            _beta = calc_log_likelihood_beta(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            _binomial = calc_log_likelihood_binomial(obs_cases, sim_cases, np.ones(self.n_steps), weights=weights, verbose=False)
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            _gamma = calc_log_likelihood_gamma(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            _negbin = calc_log_likelihood_negbin(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            _normal = calc_log_likelihood_normal(obs_cases, sim_cases, weights=weights, verbose=False)
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            _poisson = calc_log_likelihood_poisson(obs_cases, sim_cases, weights=weights, verbose=False)

        return

    def test_calc_log_likelihood_invalid_beta_inputs(self):
        return

    def test_calc_log_likelihood_invalid_binomial_inputs(self):
        return

    def test_calc_log_likelihood_invalid_gamma_inputs(self):
        return

    def test_calc_log_likelihood_invalid_negbin_inputs(self):
        return

    def test_calc_log_likelihood_invalid_normal_inputs(self):
        return

    def test_calc_log_likelihood_invalid_poisson_inputs(self):
        return


class TestThatLikelihoodTests(unittest.TestCase):
    def test_poisson_returns_NA_for_all_NA_input(self):
        ll = calc_log_likelihood_poisson(observed=np.array([np.nan, np.nan]), simulated=np.array([1, 2]), weights=None, verbose=False)
        assert np.all(np.isnan(ll))

    def test_poisson_cushions_zero_or_negative_simulated_values(self):
        ll = calc_log_likelihood_poisson(observed=np.array([1, 2]), simulated=np.array([0, -5]), weights=None, verbose=False)
        assert np.all(np.isfinite(ll))

    def test_poission_errors_on_length_mismatch(self):
        with pytest.raises(ValueError, match=re.escape("Lengths of observed (4), simulated (2), and weights (4) must all match.")):
            calc_log_likelihood_poisson(observed=np.array([1, 2, 3, 4]), simulated=np.array([1, 2]), weights=None, verbose=False)

    def test_poisson_errors_on_negative_weights(self):
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            calc_log_likelihood_poisson(observed=np.array([1, 2]), simulated=np.array([1, 2]), weights=np.array([1, -1]), verbose=False)

    def test_poisson_errors_on_zero_sum_weights(self):
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            calc_log_likelihood_poisson(observed=np.array([1, 2]), simulated=np.array([1, 2]), weights=np.array([0, 0]), verbose=False)

    def test_poisson_errors_on_non_interger_observed_values(self):
        with pytest.raises(ValueError, match=r"observed must contain non-negative integer counts for Poisson."):
            calc_log_likelihood_poisson(observed=np.array([1.5, 2]), simulated=np.array([1, 2]), weights=None, verbose=False)

    def test_poisson_warns_on_overdispersion(self):
        with pytest.warns(UserWarning, match="Var/Mean = "):
            calc_log_likelihood_poisson(observed=np.array([0, 0, 5]), simulated=np.array([1, 1, 5]), weights=None, verbose=False)

    def test_poisson_matches_manual_calculation_without_weights(self):
        observed = np.array([0, 1, 2])
        simulated = np.array([1, 2, 3])
        weights = None
        ll_manual = (observed * np.log(simulated) - simulated - gammaln(observed + 1)).sum()
        ll_func = calc_log_likelihood_poisson(observed=observed, simulated=simulated, weights=weights, verbose=False)
        assert np.isclose(ll_func, ll_manual, rtol=1e-8)

    def test_poisson_matches_manual_calculation_with_weights(self):
        observed = np.array([0, 1, 2])
        simulated = np.array([1, 2, 3])
        weights = np.array([1, 2, 0.5])
        ll_manual = (weights * (observed * np.log(simulated) - simulated - gammaln(observed + 1))).sum()
        ll_func = calc_log_likelihood_poisson(observed=observed, simulated=simulated, weights=weights, verbose=False)
        assert np.isclose(ll_func, ll_manual, rtol=1e-8)

    def test_negbin_errors_when_observed_and_simulated_lengths_differ(self):
        with pytest.raises(ValueError, match=re.escape("Lengths of observed (2), simulated (1), and weights (2) must all match.")):
            calc_log_likelihood_negbin(observed=np.array([1, 2]), simulated=np.array([1]), k=1, weights=None, verbose=False)

    def test_negbin_returns_NA_for_all_NA_input(self):
        ll = calc_log_likelihood_negbin(
            observed=np.array([np.nan, np.nan]), simulated=np.array([np.nan, np.nan]), k=1, weights=None, verbose=False
        )
        assert np.all(np.isnan(ll))

    def test_negbin_errors_on_negative_weights(self):
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            calc_log_likelihood_negbin(observed=np.array([1, 2]), simulated=np.array([1, 2]), k=1, weights=np.array([1, -1]), verbose=False)

    def test_negbin_errors_on_zero_sum_weights(self):
        with pytest.raises(ValueError, match=r"All weights are zero, cannot compute likelihood."):
            calc_log_likelihood_negbin(observed=np.array([1, 2]), simulated=np.array([1, 2]), k=1, weights=np.array([0, 0]), verbose=False)

    def test_negbin_errors_on_non_interger_observed_values(self):
        with pytest.raises(ValueError, match=r"observed must contain non-negative integer counts."):
            calc_log_likelihood_negbin(observed=np.array([1.5, 2]), simulated=np.array([1, 2]), k=1, weights=None, verbose=False)

    def test_negbin_cushions_zero_or_negative_simulated_values(self):
        ll = calc_log_likelihood_negbin(observed=np.array([1, 2]), simulated=np.array([0, -5]), k=1, weights=None, verbose=False)
        assert np.all(np.isfinite(ll))

    def test_negbin_defaults_to_poisson_when_variance_less_than_equal_mean(self):
        observed = np.array([1] * 5)
        simulated = observed.copy()
        ll_nb = calc_log_likelihood_negbin(observed=observed, simulated=simulated, k=None, weights=None, verbose=False)
        ll_pois = calc_log_likelihood_poisson(observed=observed, simulated=simulated, weights=None, verbose=False)
        assert np.isclose(ll_nb, ll_pois, rtol=1e-8)

    def test_negbin_uses_provided_k_and_warns_when_less_than_one_point_five(self):
        observed = np.array([0, 1, 2])
        simulated = np.array([1, 1, 1])
        with pytest.warns(UserWarning, match=r"k \([0-9]+\.[0-9][0-9]\) < 1.5 indicates near-Poisson dispersion."):
            _ll = calc_log_likelihood_negbin(observed=observed, simulated=simulated, k=1, weights=None, verbose=True)

    def test_negbin_matches_manual_negbin_log_likelihood_formula(self):
        observed = np.array([0, 1, 2])
        simulated = np.array([1, 2, 3])
        k = 2
        ll_vec = (
            gammaln(observed + k)
            - gammaln(k)
            - gammaln(observed + 1)
            + k * np.log(k / (k + simulated))
            + observed * np.log(simulated / (k + simulated))
        )
        ll_manual = ll_vec.sum()
        ll_func = calc_log_likelihood_negbin(observed=observed, simulated=simulated, k=k, weights=None, verbose=False)
        assert np.isclose(ll_func, ll_manual, rtol=1e-8)

    def test_negbin_matches_manual_calculation_with_weights(self):
        observed = np.array([0, 1, 2])
        simulated = np.array([1, 2, 3])
        k = 2
        weights = np.array([1, 2, 0.5])
        ll_vec = (
            gammaln(observed + k)
            - gammaln(k)
            - gammaln(observed + 1)
            + k * np.log(k / (k + simulated))
            + observed * np.log(simulated / (k + simulated))
        )
        ll_manual = (weights * ll_vec).sum()
        ll_func = calc_log_likelihood_negbin(observed=observed, simulated=simulated, k=k, weights=weights, verbose=False)
        assert np.isclose(ll_func, ll_manual, rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
