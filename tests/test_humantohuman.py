import unittest
from datetime import datetime

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.humantohuman import HumanToHuman
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.utils import sim_duration


class TestHumanToHuman(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(mods=sim_duration(datetime(2024, 1, 1), datetime(2024, 12, 31)), do_validation=False)
        iNigeria = params.location_name.index("NGA")
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        params.I_j_initial[iNigeria] = 100
        params.S_j_initial[iNigeria] -= 100

        return params, iNigeria

    # What if tau_i = 0? (no emmigration)
    def test_humantohuman_tau_i_zero(self):
        params, iNigeria = self.get_test_parameters()

        # Turn off emmigration
        params.tau_i[:] = 0

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model.run()

        for index in range(len(params.location_name)):
            if index != iNigeria:
                # No infected individuals outside Nigeria and no migration, expect Lambda to be zero
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
            else:
                # Infected individuals in Nigeria, expect Lambda to be non-zero
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."

        return

    # What if tau_i = 1? (total emmigration, no local transmission)
    def test_humantohuman_tau_i_one(self):
        params, iNigeria = self.get_test_parameters()

        # Setup no immigration into Nigeria but no local transmission due to complete emmigration
        params.tau_i[iNigeria] = 1.0

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        # Have to do this _after_ setting model components because HumanToHuman builds the pi_ij matrix
        model.patches.pi_ij[:, iNigeria] = 0.0

        model.run()

        # With no immigration and no local transmission, expect Lambda to be zero
        assert np.all(model.patches.Lambda[:, iNigeria] == 0), "HumanToHuman: tau_i = 1, Lambda_jt not zero in Nigeria."

        return

    # Special cases for pi_ij? (no connectivity)
    def test_humantohuman_pi_ij_zero(self):
        params, iNigeria = self.get_test_parameters()

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]

        # Turn off all migration, have to do this _after_ setting model components because HumanToHuman builds the pi_ij matrix
        model.patches.pi_ij *= 0.0

        model.run()

        for index in range(len(params.location_name)):
            if index != iNigeria:
                # With no infected people and no migration, expect Lambda to be zero outside Nigeria
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
            else:
                # With infected people in Nigeria, expect Lambda to be non-zero
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."

        return

    # Special cases for alpha_1?
    # Special cases for beta_j0_hum?
    # Special cases for beta_jt_human?
    # Special cases for alpha_2?


if __name__ == "__main__":
    unittest.main()
