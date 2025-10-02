import unittest

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.utils import sim_duration


class TestInfectious(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(overrides=sim_duration(), do_validation=False)
        # S - use given susceptible populations
        # E - move any exposed people back to susceptible
        params.S_j_initial += params.E_j_initial
        params.E_j_initial[:] = 0
        # I - move any infectious people back to susceptible, set I explicitly to 10,000
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 10_000
        params.S_j_initial -= params.I_j_initial
        # R - move any recovered people back to susceptible
        params.S_j_initial += params.R_j_initial
        params.R_j_initial[:] = 0
        # V1 and V2 - move any vaccinated people back to susceptible
        params.S_j_initial += params.V1_j_initial + params.V2_j_initial
        params.V1_j_initial[:] = 0
        params.V2_j_initial[:] = 0

        return params

    # Test initial distribution of infectious people - realistic sigma
    def test_initial_distribution_realistic_sigma(self):
        params = self.get_test_parameters()

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        # model.run() # don't need to run, just check initial distribution

        assert np.all(model.people.Isym[0] == np.round(model.params.sigma * model.params.I_j_initial)), (
            "I symptomatic: initial distribution not correct."
        )
        assert np.all(model.people.Iasym[0] == (params.I_j_initial - model.people.Isym[0])), "I asymptomatic: initial distribution not correct."

    # Test initial distribution of infectious people - sigma = 0
    def test_initial_distribution_sigma_zero(self):
        params = self.get_test_parameters()

        # Set sigma to 0, no symptomatic infections
        params.sigma = 0

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        # model.run() # don't need to run, just check initial distribution

        assert np.all(model.people.Isym[0] == 0), "I symptomatic: initial distribution not correct."
        assert np.all(model.people.Iasym[0] == params.I_j_initial), "I asymptomatic: initial distribution not correct."

    # Test initial distribution of infectious people - sigma = 1
    def test_initial_distribution_sigma_one(self):
        params = self.get_test_parameters()

        # Set sigma to 1, all infections are symptomatic
        params.sigma = 1

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        # model.run() # don't need to run, just check initial distribution

        assert np.all(model.people.Isym[0] == params.I_j_initial), "I symptomatic: initial distribution not correct."
        assert np.all(model.people.Iasym[0] == 0), "I asymptomatic: initial distribution not correct."

    # Steady state, d_jt = 0, mu = 0, gamma_1 = 0, gamma_2 = 0, iota = 0
    def test_infectous_steadystate(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        params.mu_jt *= 0  # turn off disease deaths
        params.gamma_1 = 0  # turn off symptomatic recovery
        params.gamma_2 = 0  # turn off asymptomatic recovery
        params.iota = 0  # turn off progression from exposed to infectious

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        model.run()

        assert np.all(model.people.Isym[-1] == model.people.Isym[0]), "I symptomatic: steady state not held."
        assert np.all(model.people.Iasym[-1] == model.people.Iasym[0]), "I asymptomatic: steady state not held."

        return

    # Non-disease deaths
    def test_infectious_non_disease_deaths(self):
        params = self.get_test_parameters()

        params.d_jt *= 10  # inflate non-disease death rate
        params.mu_jt *= 0  # turn off disease deaths
        params.gamma_1 = 0  # turn off symptomatic recovery
        params.gamma_2 = 0  # turn off asymptomatic recovery
        params.iota = 0  # turn off progression from exposed to infectious

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        model.run()

        assert np.all(model.people.Isym[-1] < model.people.Isym[0]), "I symptomatic: non-disease-deaths not occurring."
        assert np.all(model.people.Iasym[-1] < model.people.Iasym[0]), "I asymptomatic: non-disease-deaths not occurring."

        return

    # Disease deaths
    def test_infectious_disease_deaths(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off non-disease deaths
        params.mu_jt *= 10  # inflate disease death rate
        params.gamma_1 = 0  # turn off symptomatic recovery
        params.gamma_2 = 0  # turn off asymptomatic recovery
        params.iota = 0  # turn off progression from exposed to infectious

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        model.run()

        assert np.all(model.people.Isym[-1] < model.people.Isym[0]), "I symptomatic: disease deaths not occurring."
        assert np.all(model.people.Iasym[-1] == model.people.Iasym[0]), "I asymptomatic: disease deaths occurring in the asymptomatic population."

        return

    # Symptomatic Recovery
    def test_infectious_symptomatic_recovery(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off non-disease deaths
        params.mu_jt *= 0  # turn off disease deaths
        # leave this on: params.gamma_1 = 0 # turn off symptomatic recovery
        params.gamma_2 = 0  # turn off asymptomatic recovery
        params.iota = 0  # turn off progression from exposed to infectious

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        model.run()

        assert np.all(model.people.Isym[-1] < model.people.Isym[0]), "I symptomatic: recovery not occurring."
        assert np.all(model.people.Iasym[-1] == model.people.Iasym[0]), "I asymptomatic: recovery occurring in the asymptomatic population."

        return

    # Asymptomatic Recovery
    def test_infectious_asymptomatic_recovery(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off non-disease deaths
        params.mu_jt *= 0  # turn off disease deaths
        params.gamma_1 = 0  # turn off symptomatic recovery
        # leave this on: params.gamma_2 = 0 # turn off asymptomatic recovery
        params.iota = 0  # turn off progression from exposed to infectious

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        model.run()

        assert np.all(model.people.Isym[-1] == model.people.Isym[0]), "I symptomatic: recovery occurring in the symptomatic population."
        assert np.all(model.people.Iasym[-1] < model.people.Iasym[0]), "I asymptomatic: recovery not occurring."

        return

    # Newly infectious
    def test_infectious_newly_infectious(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off non-disease deaths
        params.mu_jt *= 0  # turn off disease deaths
        params.gamma_1 = 0  # turn off symptomatic recovery
        params.gamma_2 = 0  # turn off asymptomatic recovery
        # leave this on: params.iota = 0 # turn off progression from exposed to infectious
        # Move 10,000 people from susceptible into exposed
        params.E_j_initial[:] = 10_000
        params.S_j_initial -= params.E_j_initial

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census]
        model.run()

        assert np.all(model.people.Isym[-1] > model.people.Isym[0]), "I symptomatic: progression from exposed to infectious not occurring."
        assert np.all(model.people.Iasym[-1] > model.people.Iasym[0]), "I asymptomatic: progression from exposed to infectious not occurring."

        return


if __name__ == "__main__":
    unittest.main()
