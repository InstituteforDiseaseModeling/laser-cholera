import unittest
from datetime import datetime

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.metapop.vaccinated import Vaccinated
from laser_cholera.utils import sim_duration


class TestVaccinated(unittest.TestCase):
    @staticmethod
    def get_test_parameters(V1=20_000, V2=10_000, overrides=None):
        if overrides is None:
            overrides = sim_duration()
        params = get_parameters(overrides=overrides, do_validation=False)
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
        # V1 and V2 - move any vaccinated people back to susceptible, set explicitly to 20,000 and 10,000 respectively
        params.S_j_initial += params.V1_j_initial + params.V2_j_initial
        params.V1_j_initial[:] = V1
        params.V2_j_initial[:] = V2
        params.S_j_initial -= params.V1_j_initial + params.V2_j_initial

        return params

    # Test initial distribution of vaccinated people - realistic phi_1 and phi_2
    def test_initial_distribution_realistic_phi(self):
        params = self.get_test_parameters()

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution

        assert np.all(model.people.V1imm[0] == np.round(model.params.phi_1 * model.params.V1_j_initial)), "V1_imm: initial distribution not correct."
        assert np.all(model.people.V1sus[0] == (model.params.V1_j_initial - model.people.V1imm[0])), "V1_sus: initial distribution not correct."
        assert np.all(model.people.V1inf[0] == 0), "V1_inf: initial distribution not correct."
        assert np.all(model.patches.V1[0] == model.params.V1_j_initial), "V1: initial distribution not correct."
        assert np.all(model.people.V2imm[0] == np.round(model.params.phi_2 * model.params.V2_j_initial)), "V2_imm: initial distribution not correct."
        assert np.all(model.people.V2sus[0] == (model.params.V2_j_initial - model.people.V2imm[0])), "V2_sus: initial distribution not correct."
        assert np.all(model.people.V2inf[0] == 0), "V2_inf: initial distribution not correct."
        assert np.all(model.patches.V2[0] == model.params.V2_j_initial), "V2: initial distribution not correct."

    # Test initial distribution of vaccinated people - phi_1 and phi_2 = 0
    def test_initial_distribution_phi_zero(self):
        params = self.get_test_parameters()

        params.phi_1 = 0
        params.phi_2 = 0

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution

        assert np.all(model.people.V1imm[0] == 0), "V1_imm: initial distribution not correct."
        assert np.all(model.people.V1sus[0] == model.params.V1_j_initial), "V1_sus: initial distribution not correct."
        assert np.all(model.people.V1inf[0] == 0), "V1_inf: initial distribution not correct."
        assert np.all(model.patches.V1[0] == model.params.V1_j_initial), "V1: initial distribution not correct."
        assert np.all(model.people.V2imm[0] == 0), "V2_imm: initial distribution not correct."
        assert np.all(model.people.V2sus[0] == model.params.V2_j_initial), "V2_sus: initial distribution not correct."
        assert np.all(model.people.V2inf[0] == 0), "V2_inf: initial distribution not correct."
        assert np.all(model.patches.V2[0] == model.params.V2_j_initial), "V2: initial distribution not correct."

    # Test initial distribution of vaccinated people - phi_1 and phi_2 = 1
    def test_initial_distribution_phi_one(self):
        params = self.get_test_parameters()

        params.phi_1 = 1
        params.phi_2 = 1

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial distribution

        assert np.all(model.people.V1imm[0] == model.params.V1_j_initial), "V1_imm: initial distribution not correct."
        assert np.all(model.people.V1sus[0] == 0), "V1_sus: initial distribution not correct."
        assert np.all(model.people.V1inf[0] == 0), "V1_inf: initial distribution not correct."
        assert np.all(model.patches.V1[0] == model.params.V1_j_initial), "V1: initial distribution not correct."
        assert np.all(model.people.V2imm[0] == model.params.V2_j_initial), "V2_imm: initial distribution not correct."
        assert np.all(model.people.V2sus[0] == 0), "V2_sus: initial distribution not correct."
        assert np.all(model.people.V2inf[0] == 0), "V2_inf: initial distribution not correct."
        assert np.all(model.patches.V2[0] == model.params.V2_j_initial), "V2: initial distribution not correct."

    # Steady state
    def test_vaccinated_steadystate(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] == model.people.V1imm[0]), "V1 immunized: steady state not held."
        assert np.all(model.people.V1sus[-1] == model.people.V1sus[0]), "V1 susceptible: steady state not held."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] == model.patches.V1[0]), "V1 total: steady state not held."
        assert np.all(model.people.V2imm[-1] == model.people.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.people.V2sus[-1] == model.people.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] == model.patches.V2[0]), "V2 total: steady state not held."

        return

    # Non-disease mortality
    def test_vaccinated_non_disease_deaths(self):
        params = self.get_test_parameters()

        params.d_jt *= 10  # inflate non-disease death rate
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] < model.people.V1imm[0]), "V1 immunized: non-disease deaths not occurring."
        assert np.all(model.people.V1sus[-1] < model.people.V1sus[0]), "V1 susceptible: non-disease deaths not occurring."
        # There is no infection, so V1inf == 0
        # assert np.all(model.people.V1inf[-1] < model.people.V1inf[0]), "V1 infected: non-disease deaths not occurring."
        assert np.all(model.patches.V1[-1] < model.patches.V1[0]), "V1 total: non-disease deaths not occurring."
        assert np.all(model.people.V2imm[-1] < model.people.V2imm[0]), "V2 immunized: non-disease deaths not occurring."
        assert np.all(model.people.V2sus[-1] < model.people.V2sus[0]), "V2 susceptible: non-disease deaths not occurring."
        # There is no infection, so V2inf == 0
        # assert np.all(model.people.V2inf[-1] < model.people.V2inf[0]), "V2 infected: non-disease deaths not occurring."
        assert np.all(model.patches.V2[-1] < model.patches.V2[0]), "V2 total: non-disease deaths not occurring."

        return

    # Waning immunity - one dose
    def test_vaccinated_waning_one_dose_immunity(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        # leave this on: params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] < model.people.V1imm[0]), "V1 immunized: missing waning immunity."
        assert np.all(model.people.V1sus[-1] > model.people.V1sus[0]), "V1 susceptible: missing waning immunity."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: should not change."
        assert np.all(model.patches.V1[-1] == model.patches.V1[0]), "V1 total: should not change."
        assert np.all(model.people.V2imm[-1] == model.people.V2imm[0]), "V2 immunized: should not change."
        assert np.all(model.people.V2sus[-1] == model.people.V2sus[0]), "V2 susceptible: should not change."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: should not change."
        assert np.all(model.patches.V2[-1] == model.patches.V2[0]), "V2 total: should not change."

        return

    # Waning immunity - two doses
    def test_vaccinated_waning_two_dose_immunity(self):
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        # leave this on: params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] == model.people.V1imm[0]), "V1 immunized: should not change."
        assert np.all(model.people.V1sus[-1] == model.people.V1sus[0]), "V1 susceptible: should not change."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: should not change."
        assert np.all(model.patches.V1[-1] == model.patches.V1[0]), "V1 total: should not change."
        assert np.all(model.people.V2imm[-1] < model.people.V2imm[0]), "V2 immunized: missing waning immunity."
        assert np.all(model.people.V2sus[-1] > model.people.V2sus[0]), "V2 susceptible: missing waning immunity."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: should not change."
        assert np.all(model.patches.V2[-1] == model.patches.V2[0]), "V2 total: should not change."

        return

    # Newly vaccinated - one dose, realistic efficacy
    def test_vaccinated_one_dose_vaccination_realistic_efficacy(self):
        params = self.get_test_parameters(V1=0, V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt[0:32] = 100
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] > model.people.V1imm[0]), "V1 immunized: missing newly vaccinated people."
        assert np.all(model.people.V1sus[-1] > model.people.V1sus[0]), "V1 susceptible: missing newly vaccinated people."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] > model.patches.V1[0]), "V1 total: missing newly vaccinated people."
        assert np.all(model.people.V2imm[-1] == model.people.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.people.V2sus[-1] == model.people.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] == model.patches.V2[0]), "V2 total: steady state not held."

        return

    # Newly vaccinated - one dose, perfect efficacy
    def test_vaccinated_one_dose_vaccination_perfect_efficacy(self):
        params = self.get_test_parameters(V1=0, V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt[0:32] = 100
        params.nu_2_jt *= 0  # turn off second dose vaccination
        params.phi_1 = 1.0

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] > model.people.V1imm[0]), "V1 immunized: missing newly vaccinated people."
        assert np.all(model.people.V1sus[-1] == model.people.V1sus[0]), "V1 susceptible: should not have newly vaccinated people."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] > model.patches.V1[0]), "V1 total: missing newly vaccinated people."
        assert np.all(model.people.V2imm[-1] == model.people.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.people.V2sus[-1] == model.people.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] == model.patches.V2[0]), "V2 total: steady state not held."

        return

    # Newly vaccinated - one dose, no efficacy
    def test_vaccinated_one_dose_vaccination_no_efficacy(self):
        params = self.get_test_parameters(V1=0, V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt[0:32] = 100
        params.nu_2_jt *= 0  # turn off second dose vaccination
        params.phi_1 = 0.0

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] == model.people.V1imm[0]), "V1 immunized: should not have newly vaccinated people."
        assert np.all(model.people.V1sus[-1] > model.people.V1sus[0]), "V1 susceptible: missing newly vaccinated people."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] > model.patches.V1[0]), "V1 total: missing newly vaccinated people."
        assert np.all(model.people.V2imm[-1] == model.people.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.people.V2sus[-1] == model.people.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] == model.patches.V2[0]), "V2 total: steady state not held."

        return

    # Newly vaccinated - two doses, realistic efficacy
    def test_vaccinated_two_dose_vaccination_realistic_efficacy(self):
        params = self.get_test_parameters(V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt[0:32] = 100

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] < model.people.V1imm[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.people.V1sus[-1] < model.people.V1sus[0]), "V1 susceptible: should be missing newly vaccinated people."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] < model.patches.V1[0]), "V1 total: should be missing newly vaccinated people."
        assert np.all(model.people.V2imm[-1] > model.people.V2imm[0]), "V2 immunized: missing newly vaccinated people."
        assert np.all(model.people.V2sus[-1] > model.people.V2sus[0]), "V2 susceptible: missing newly vaccinated people."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] > model.patches.V2[0]), "V2 total: missing newly vaccinated people."

        return

    # Newly vaccinated - two doses, perfect efficacy
    def test_vaccinated_two_dose_vaccination_perfect_efficacy(self):
        params = self.get_test_parameters(V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt[0:32] = 100
        params.phi_2 = 1.0

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] < model.people.V1imm[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.people.V1sus[-1] < model.people.V1sus[0]), "V1 susceptible: should be missing newly vaccinated people."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] < model.patches.V1[0]), "V1 total: should be missing newly vaccinated people."
        assert np.all(model.people.V2imm[-1] > model.people.V2imm[0]), "V2 immunized: missing newly vaccinated people."
        assert np.all(model.people.V2sus[-1] == model.people.V2sus[0]), "V2 susceptible: steady state not held."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] > model.patches.V2[0]), "V2 total: missing newly vaccinated people."

        return

    # Newly vaccinated - two doses, no efficacy
    def test_vaccinated_two_dose_vaccination_no_efficacy(self):
        params = self.get_test_parameters(V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt[0:32] = 100
        params.phi_2 = 0.0

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1imm[-1] < model.people.V1imm[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.people.V1sus[-1] < model.people.V1sus[0]), "V1 susceptible: should be missing newly vaccinated people."
        assert np.all(model.people.V1inf[-1] == model.people.V1inf[0]), "V1 infected: steady state not held."
        assert np.all(model.patches.V1[-1] < model.patches.V1[0]), "V1 total: should be missing newly vaccinated people."
        assert np.all(model.people.V2imm[-1] == model.people.V2imm[0]), "V2 immunized: steady state not held."
        assert np.all(model.people.V2sus[-1] > model.people.V2sus[0]), "V2 susceptible: missing newly vaccinated people."
        assert np.all(model.people.V2inf[-1] == model.people.V2inf[0]), "V2 infected: steady state not held."
        assert np.all(model.patches.V2[-1] > model.patches.V2[0]), "V2 total: missing newly vaccinated people."

        return

    def test_doses_given_against_dose_schedule(self):
        params = self.get_test_parameters(V1=0, V2=0, overrides=sim_duration(start=datetime(2023, 1, 1), stop=datetime(2024, 12, 17)))

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]
        model.run()

        nonzero_param_indices = np.nonzero(model.params.nu_1_jt)
        if len(nonzero_param_indices[0]) > 0:
            # Check any non-zero days in nu_1_jt should have non-zero doses in dose_one_doses
            assert np.all(model.patches.dose_one_doses[nonzero_param_indices] != 0), (
                "All non-zero nu_1_jt should have non-zero doses in dose_one_doses."
            )
            # Check that first doses given match the schedule
            nonzero_dose_indices = np.nonzero(model.patches.dose_one_doses)
            assert np.all(model.params.nu_1_jt[nonzero_dose_indices] != 0), (
                "All non-zero dose_one_doses should have non-zero doses in params.nu_1_jt."
            )

        nonzero_param_indices = np.nonzero(model.params.nu_2_jt)
        if len(nonzero_param_indices[0]) > 0:
            # Check any non-zero days in nu_2_jt should have non-zero doses in dose_two_doses
            assert np.all(model.patches.dose_two_doses[nonzero_param_indices] != 0), (
                "All non-zero nu_2_jt should have non-zero doses in dose_two_doses."
            )
            # Check that second doses given match the schedule
            nonzero_dose_indices = np.nonzero(model.patches.dose_two_doses)
            assert np.all(model.params.nu_2_jt[nonzero_dose_indices] != 0), (
                "All non-zero dose_two_doses should have non-zero doses in params.nu_2_jt."
            )

        return


if __name__ == "__main__":
    unittest.main()
