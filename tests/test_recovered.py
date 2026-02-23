import unittest

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible  # we need this to put waning recovereds back into susceptibles
from laser_cholera.utils import sim_duration


class TestRecovered(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(mods=sim_duration(), do_validation=False)
        # S - use given susceptible populations
        # E - move any exposed people back to susceptible
        params.S_j_initial += params.E_j_initial
        params.E_j_initial[:] = 0
        # I - move any infectious people back to susceptible
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        # R - move any recovered people back to susceptible and set explicitly to 50,000
        params.S_j_initial += params.R_j_initial
        params.R_j_initial[:] = 50_000
        params.S_j_initial -= params.R_j_initial
        # V1 and V2 - move any vaccinated people back to susceptible
        params.S_j_initial += params.V1_j_initial + params.V2_j_initial
        params.V1_j_initial[:] = 0
        params.V2_j_initial[:] = 0

        return params

    def test_recovered_steadystate(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        params.epsilon = 0  # turn off waning natural immunity

        model = Model(parameters=params)
        model.components = [Susceptible, Recovered, Census]
        model.run()

        assert np.all(model.people.R[-1] == model.people.R[0]), "Recovered: steady state not held."

        return

    def test_recovered_deaths(self):
        params = self.get_test_parameters()

        model = Model(parameters=params)
        model.components = [Susceptible, Recovered, Census]
        model.run()

        assert np.all(model.people.R[-1] < model.people.R[0]), "Recovered: deaths not occurring."

        return

    def test_recovered_waning_immunity(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths

        model = Model(parameters=params)
        model.components = [Susceptible, Recovered, Census]
        model.run()

        assert np.all(model.people.R[-1] < model.people.R[0]), "Recovered: waning immunity not occurring."

        return


if __name__ == "__main__":
    unittest.main()
