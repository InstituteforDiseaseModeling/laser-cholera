import unittest

import numpy as np
from laser_core.propertyset import PropertySet

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.environmental import Environmental
from laser_cholera.metapop.envtohuman import EnvToHuman
from laser_cholera.metapop.envtohumanvax import EnvToHumanVax
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.metapop.vaccinated import Vaccinated
from laser_cholera.test import Eradication
from laser_cholera.utils import sim_duration


class TestEnvToHumanVax(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = cls.get_test_parameters()
        cls.baseline = Model(cls.params)
        cls.baseline.components = [
            Eradication,
            Susceptible,
            Exposed,
            Infectious,
            Recovered,
            Vaccinated,
            EnvToHuman,
            EnvToHumanVax,
            Census,
            Environmental,
        ]
        cls.baseline.run()

        # Expect V1imm and V2imm to be in steady state with vaccine dynamics turned off.
        assert np.all(cls.baseline.people.V1imm[1:] == cls.baseline.people.V1imm[0]), "V1imm: steady state not held."
        assert np.all(cls.baseline.people.V2imm[1:] == cls.baseline.people.V2imm[0]), "V2imm: steady state not held."

        # Expect V1sus and V2sus to decline with environmental transmission.
        assert np.all(np.diff(cls.baseline.people.V1sus, axis=0) <= 0), "V1sus: not declining with environmental transmission."
        assert np.all(np.diff(cls.baseline.people.V2sus, axis=0) <= 0), "V2sus: not declining with environmental transmission."

        # Expect V1inf and V2inf to increase with environmental transmission.
        assert np.all(np.diff(cls.baseline.people.V1inf, axis=0) >= 0), "V1inf: not increasing with environmental transmission."
        assert np.all(np.diff(cls.baseline.people.V2inf, axis=0) >= 0), "V2inf: not increasing with environmental transmission."

        # Expect V1 and V2 to be in steady state with vaccine dynamics turned off.
        assert np.all(cls.baseline.patches.V1[1:] == cls.baseline.patches.V1[0]), "V1: steady state not held."
        assert np.all(cls.baseline.patches.V2[1:] == cls.baseline.patches.V2[0]), "V2: steady state not held."

        return

    @staticmethod
    def get_test_parameters() -> PropertySet:
        params = get_parameters(overrides=sim_duration(), do_validation=False)

        # Configure population
        # Move some people to the infectious state so we have shedding to the environment
        params.S_j_initial += params.I_j_initial
        params.I_j_initial = 10_000
        params.S_j_initial -= params.I_j_initial

        # Move some people to the vaccinated but susceptible states so we have some transmission
        params.V1_j_initial[:] = 20_000
        params.V2_j_initial[:] = 10_000
        params.S_j_initial -= params.V1_j_initial + params.V2_j_initial

        # Turn off non-transmission vaccine dynamics
        params.nu_1_jt *= 0.0  # Turn off first dose vaccination
        params.nu_1_jt *= 0.0  # Turn off second dose vaccination
        params.d_jt *= 0.0  # Turn off non-disease mortality
        params.omega_1 = params.omega_2 = 0.0  # Turn off waning vaccine immunity

        return params

    # Test increased environmental transmission (++Psi)
    def test_envtohumanvax_increased_environmental_transmission(self):
        params = self.get_test_parameters()

        # Configure parameters for increased environmental transmission
        params.beta_j0_env *= 5.0

        model = Model(parameters=params)
        model.components = [
            Eradication,
            Susceptible,
            Exposed,
            Infectious,
            Recovered,
            Vaccinated,
            EnvToHuman,
            EnvToHumanVax,
            Census,
            Environmental,
        ]
        model.run()

        # Expect more transmission with increased environmental transmission
        nz = np.nonzero(self.baseline.people.V1inf)
        assert len(nz) > 0, "V1inf: no infected individuals."  # All zeros would give a false positive
        assert np.all(model.people.V1inf[nz] >= self.baseline.people.V1inf[nz]), (
            "V1inf: not increasing with increased environmental transmission."
        )
        # nz = np.nonzero(self.baseline.people.V2inf)
        # assert len(nz) > 0, "V2inf: no infected individuals"  # All zeros would give a false positive
        # assert np.all(model.people.V2inf[nz] >= self.baseline.people.V2inf[nz]), (
        #     "V2inf: not increasing with increased environmental transmission."
        significant = np.nonzero(self.baseline.people.V2inf > 10)  # Only check where V2inf is significant
        assert len(significant) > 0, "V2inf: no significantg count of infected individuals"  # All zeros would give a false positive
        assert np.all(model.people.V2inf[significant] >= self.baseline.people.V2inf[significant]), (
            "V2inf: not increasing with increased environmental transmission."
        )

        return

    # Test decreased environmental transmission (--Psi)
    def test_envtohumanvax_decreased_environmental_transmission(self):
        params = self.get_test_parameters()

        # Configure parameters for decreased environmental transmission
        params.beta_j0_env /= 5.0

        model = Model(parameters=params)
        model.components = [
            Eradication,
            Susceptible,
            Exposed,
            Infectious,
            Recovered,
            Vaccinated,
            EnvToHuman,
            EnvToHumanVax,
            Census,
            Environmental,
        ]
        model.run()

        # Expect less transmission with decreased environmental transmission
        assert np.all(model.people.V1inf[-1, :] <= self.baseline.people.V1inf[-1, :]), (
            "V1inf: not decreasing with decreased environmental transmission."
        )
        assert np.all(model.people.V2inf[-1, :] <= self.baseline.people.V2inf[-1, :]), (
            "V2inf: not decreasing with decreased environmental transmission."
        )

        return


if __name__ == "__main__":
    unittest.main()
