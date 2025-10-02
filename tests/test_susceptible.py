import unittest
from datetime import datetime

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.utils import sim_duration


class TestSusceptible(unittest.TestCase):
    @staticmethod
    def get_test_parameters(overrides=None):
        params = get_parameters(overrides=overrides if overrides else sim_duration(), do_validation=False)
        # S - use given susceptible populations
        # E - move any exposed people back to susceptible
        params.S_j_initial += params.E_j_initial
        params.E_j_initial[:] = 0
        # I - move any infectious people back to susceptible
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        # R - move any recovered people back to susceptible
        params.S_j_initial += params.R_j_initial
        params.R_j_initial[:] = 0
        # V1 and V2 - move any vaccinated people back to susceptible
        params.S_j_initial += params.V1_j_initial + params.V2_j_initial
        params.V1_j_initial[:] = 0
        params.V2_j_initial[:] = 0

        return params

    def test_susceptible_steadystate(self):
        params = self.get_test_parameters()

        params.b_jt *= 0  # turn off births
        params.d_jt *= 0  # turn off deaths

        model = Model(parameters=params)
        model.components = [Susceptible, Census]
        model.run()

        assert np.all(model.people.S[-1] == model.people.S[0]), "Susceptible: steady state not held."

        return

    def test_susceptible_vitalstatistics(self):
        params = self.get_test_parameters()

        params.b_jt *= 5  # turn up births
        params.d_jt *= 5  # turn up deaths

        model = Model(parameters=params)
        model.components = [Susceptible, Census]
        model.run()

        delta = model.patches.births[:-1] - model.patches.non_disease_deaths[:-1]
        assert np.all(delta == (model.people.S[1:] - model.people.S[:-1])), "Susceptible: births - deaths != change in susceptible population."

        return

    def test_susceptible_deaths(self):
        params = self.get_test_parameters()

        params.b_jt *= 0  # turn off births

        model = Model(parameters=params)
        model.components = [Susceptible, Census]
        model.run()

        assert np.all(model.people.S[-1] < model.people.S[0]), "Susceptible: deaths not occurring."
        assert np.all(model.patches.non_disease_deaths[:-1] == (model.people.S[:-1] - model.people.S[1:])), (
            "Susceptible: deaths not recorded correctly."
        )

        return

    def test_susceptible_births(self):
        params = self.get_test_parameters(overrides={"date_start": datetime(2025, 3, 24), "date_stop": datetime(2025, 4, 25), "nticks": 32})

        params.d_jt *= 0  # turn off deaths

        model = Model(parameters=params)
        model.components = [Susceptible, Census]
        model.run()

        assert np.all(model.people.S[-1] > model.people.S[0]), "Susceptible: births not occurring."
        assert np.all(model.patches.births[:-1] == (model.people.S[1:] - model.people.S[:-1])), "Susceptible: births not recorded correctly."

        return


if __name__ == "__main__":
    unittest.main()
