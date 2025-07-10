import unittest
from datetime import datetime

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.humantohuman import HumanToHuman
from laser_cholera.metapop.humantohumanvax import HumanToHumanVax
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.metapop.vaccinated import Vaccinated
from laser_cholera.utils import sim_duration


class TestHumanToHumanVax(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(overrides=sim_duration(datetime(2024, 1, 1), datetime(2024, 12, 31)), do_validation=False)
        iNigeria = params.location_name.index("NGA")

        # S - use given values
        # E - move any exposed back into S
        params.S_j_initial += params.E_j_initial
        params.E_j_initial[:] = 0
        # I - move any infectious back into S, set I in Nigeria to 100
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        params.I_j_initial[iNigeria] = 100
        params.S_j_initial[iNigeria] -= params.I_j_initial[iNigeria]
        # R - move any recovered back into S
        params.S_j_initial += params.R_j_initial
        params.R_j_initial[:] = 0
        # V (V1 and V2) - move V1 and V2 back into S, then set explicitly to 20,000 and 10,000 respectively
        params.S_j_initial += params.V1_j_initial + params.V2_j_initial
        params.V1_j_initial[:] = 20_000
        params.V2_j_initial[:] = 10_000
        params.S_j_initial -= params.V1_j_initial + params.V2_j_initial

        # turn off non-disease related mortality
        params.d_jt *= 0.0

        # turn off waning vaccine immunity
        params.omega_1 = params.omega_2 = 0.0

        # turn off vaccination campaigns
        params.nu_1_jt *= 0.0
        params.nu_2_jt *= 0.0

        return params, iNigeria

    # What if tau_i = 0? (no emmigration)
    def test_humantohuman_tau_i_zero(self):
        params, iNigeria = self.get_test_parameters()

        # Turn off migration
        params.tau_i *= 0.0

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Vaccinated, Census, HumanToHuman, HumanToHumanVax]

        # Have to do this _after_ setting model components because HumanToHuman builds the pi_ij matrix
        model.patches.pi_ij *= 0.0

        model.run()
        for index in range(len(params.location_name)):
            if index != iNigeria:
                # With no infectious people and no migration, expect Lambda to be zero
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
                # Expect all vaccine populations to be in steady state
                assert np.all(model.people.V1imm[1:, index] == model.people.V1imm[0, index]), "V1imm: steady state not held."
                assert np.all(model.people.V2imm[1:, index] == model.people.V2imm[0, index]), "V2imm: steady state not held."
                assert np.all(model.people.V1sus[1:, index] == model.people.V1sus[0, index]), "V1sus: steady state not held."
                assert np.all(model.people.V2sus[1:, index] == model.people.V2sus[0, index]), "V2sus: steady state not held."
                assert np.all(model.people.V1inf[1:, index] == model.people.V1inf[0, index]), "V1inf: steady state not held."
                assert np.all(model.people.V2inf[1:, index] == model.people.V2inf[0, index]), "V2inf: steady state not held."
                assert np.all(model.patches.V1[1:, index] == model.patches.V1[0, index]), "V1: steady state not held."
                assert np.all(model.patches.V2[1:, index] == model.patches.V2[0, index]), "V2: steady state not held."
            else:
                # With infectious people in Nigeria, expect Lambda to be non-zero
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."
                # Expect immune vaccine populations to be in steady state
                assert np.all(model.people.V1imm[1:, index] == model.people.V1imm[0, index]), "V1imm: steady state not held."
                assert np.all(model.people.V2imm[1:, index] == model.people.V2imm[0, index]), "V2imm: steady state not held."
                # Expect susceptible vaccine populations to decline
                assert np.all(np.diff(model.people.V1sus, axis=0) <= 0), "V1sus: not declining with human to human transmission."
                assert np.all(np.diff(model.people.V2sus, axis=0) <= 0), "V2sus: not declining with human to human transmission."
                # Expect infected vaccine populations to increase
                assert np.all(np.diff(model.people.V1inf, axis=0) >= 0), "V1inf: not increasing with human to human transmission."
                assert np.all(np.diff(model.people.V2inf, axis=0) >= 0), "V2inf: not increasing with human to human transmission."
                # Expect total vaccine populations to be in steady state
                assert np.all(model.patches.V1[1:, index] == model.patches.V1[0, index]), "V1: steady state not held."
                assert np.all(model.patches.V2[1:, index] == model.patches.V2[0, index]), "V2: steady state not held."

        return


if __name__ == "__main__":
    unittest.main()
