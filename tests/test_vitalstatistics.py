import unittest
from datetime import datetime

import numpy as np

from laser.cholera.metapop.census import Census
from laser.cholera.metapop.exposed import Exposed
from laser.cholera.metapop.infectious import Infectious
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.recovered import Recovered
from laser.cholera.metapop.susceptible import Susceptible
from laser.cholera.metapop.vaccinated import Vaccinated
from laser.cholera.utils import sim_duration


class TestVitalStatistics(unittest.TestCase):
    @staticmethod
    def get_test_parameters(overrides=None):
        params = get_parameters(mods=overrides if overrides else sim_duration(), do_validation=False)
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

        # Distribute population across various states to our liking
        params.E_j_initial[:] = np.random.binomial(params.S_j_initial, 0.5).astype(params.E_j_initial.dtype)  # 50% of S become E
        params.S_j_initial -= params.E_j_initial

        params.I_j_initial[:] = np.random.binomial(params.S_j_initial, 0.1).astype(params.I_j_initial.dtype)  # 10% of remaining S become I
        params.S_j_initial -= params.I_j_initial

        params.R_j_initial[:] = np.random.binomial(params.S_j_initial, 0.5).astype(params.R_j_initial.dtype)  # 50% of remaining S become R
        params.S_j_initial -= params.R_j_initial

        params.V1_j_initial[:] = np.random.binomial(params.S_j_initial, 0.1).astype(params.V1_j_inital.dtype)  # 10% of remaining S become V1
        params.S_j_initial -= params.V1_j_initial

        params.V2_j_initial[:] = np.random.binomial(params.S_j_initial, 0.1).astype(params.V2_j_initial.dtype)  # 10% of remaining S become V2
        params.S_j_initial -= params.V2_j_initial

        return params

    def test_susceptible_steadystate(self):
        params = self.get_test_parameters()

        params.b_jt *= 0  # turn off births
        params.d_jt *= 0  # turn off deaths
        params.iota *= 0  # turn off progression from E to I
        params.mu_j_baseline *= 0  # turn off disease deaths
        params.gamma_1 *= 0  # turn off recovery of symptomatic (Isym to R)
        params.gamma_2 *= 0  # turn off recovery of asymptomatic (Iasym to R)
        params.epsilon *= 0  # turn off waning natural immunity (R to S)
        params.omega_1 *= 0  # turn off waning vaccine immunity (V1 to S)
        params.omega_2 *= 0  # turn off waning vaccine immunity (V2 to S)
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Recovered, Infectious, Vaccinated, Census]
        model.run()

        assert np.all(model.people.S[-1] == model.people.S[0]), "Susceptible: steady state not held."
        assert np.all(model.people.E[-1] == model.people.E[0]), "Exposed: steady state not held."
        assert np.all(model.people.Isym[-1] == model.people.Isym[0]), "Infectious (symptomatic): steady state not held."
        assert np.all(model.people.Iasym[-1] == model.people.Iasym[0]), "Infectious (asymptomatic): steady state not held."
        assert np.all(model.people.R[-1] == model.people.R[0]), "Recovered: steady state not held."
        assert np.all(model.people.V1[-1] == model.people.V1[0]), "Vaccinated (one dose, immune): steady state not held."

        assert np.all(model.people.V2[-1] == model.people.V2[0]), "Vaccinated (two doses, immune): steady state not held."

        assert np.all(model.patches.births == 0), "Susceptible: non-zero births with b_jt = 0."
        assert np.all(model.patches.non_disease_deaths == 0), "Susceptible: non-zero non-disease deaths with d_jt = 0."

        return

    def test_vitalstatistics_without_disease_mortality(self):
        params = self.get_test_parameters()

        params.b_jt *= 5  # turn _up_ births
        params.d_jt *= 5  # turn _up_ deaths

        params.iota *= 0  # turn off progression from E to I
        params.mu_j_baseline *= 0  # turn off disease deaths
        params.gamma_1 *= 0  # turn off recovery of symptomatic (Isym to R)
        params.gamma_2 *= 0  # turn off recovery of asymptomatic (Iasym to R)
        params.epsilon *= 0  # turn off waning natural immunity (R to S)
        params.omega_1 *= 0  # turn off waning vaccine immunity (V1 to S)
        params.omega_2 *= 0  # turn off waning vaccine immunity (V2 to S)
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Recovered, Infectious, Vaccinated, Census]
        model.run()

        assert np.all(model.patches.births >= 0), "Susceptible: missing births with b_jt > 0."
        assert np.all(model.patches.non_disease_deaths >= 0), "Susceptible: missing non-zero non-disease deaths with d_jt > 0."
        aggregate = model.people.S + model.people.E + model.people.Isym + model.people.Iasym + model.people.R + model.people.V1 + model.people.V2
        delta = model.patches.births[:-1] - model.patches.non_disease_deaths[:-1]
        assert np.all(delta == (aggregate[1:] - aggregate[:-1])), "Susceptible: births - deaths != change in susceptible population."

        return

    def test_vitalstatistics_with_disease_mortality(self):
        params = self.get_test_parameters()

        params.b_jt *= 5  # turn _up_ births
        params.d_jt *= 5  # turn _up_ deaths
        params.mu_j_baseline *= 5  # turn _up_ disease deaths

        params.iota *= 0  # turn off progression from E to I
        params.gamma_1 *= 0  # turn off recovery of symptomatic (Isym to R)
        params.gamma_2 *= 0  # turn off recovery of asymptomatic (Iasym to R)
        params.epsilon *= 0  # turn off waning natural immunity (R to S)
        params.omega_1 *= 0  # turn off waning vaccine immunity (V1 to S)
        params.omega_2 *= 0  # turn off waning vaccine immunity (V2 to S)
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Recovered, Infectious, Vaccinated, Census]
        model.run()

        assert np.all(model.patches.births >= 0), "Susceptible: missing births with b_jt > 0."
        assert np.all(model.patches.non_disease_deaths >= 0), "Susceptible: missing non-zero non-disease deaths with d_jt > 0."
        aggregate = model.people.S + model.people.E + model.people.Isym + model.people.Iasym + model.people.R + model.people.V1 + model.people.V2
        delta = model.patches.births - model.patches.non_disease_deaths - model.patches.disease_deaths
        assert np.all(delta[:-1] == (aggregate[1:] - aggregate[:-1])), "Susceptible: births - deaths != change in susceptible population."

        return

    def test_births(self):
        params = self.get_test_parameters(overrides={"date_start": datetime(2025, 3, 24), "date_stop": datetime(2025, 4, 25), "nticks": 32})

        params.d_jt *= 0  # turn off non-disease deaths
        params.mu_j_baseline *= 0  # turn off disease deaths

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Recovered, Infectious, Vaccinated, Census]
        model.run()

        aggregate = model.people.S + model.people.E + model.people.Isym + model.people.Iasym + model.people.R + model.people.V1 + model.people.V2
        assert np.all(model.patches.births[:-1] == (aggregate[1:] - aggregate[:-1])), "total population: births not recorded correctly."

        return

    def test_deaths(self):
        params = self.get_test_parameters()

        params.b_jt *= 0  # turn off births
        params.iota *= 0  # turn off progression from E to I
        params.mu_j_baseline *= 0  # turn off disease deaths
        params.gamma_1 *= 0  # turn off recovery of symptomatic (Isym to R)
        params.gamma_2 *= 0  # turn off recovery of asymptomatic (Iasym to R)
        params.epsilon *= 0  # turn off waning natural immunity (R to S)
        params.omega_1 *= 0  # turn off waning vaccine immunity (V1 to S)
        params.omega_2 *= 0  # turn off waning vaccine immunity (V2 to S)
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Recovered, Infectious, Vaccinated, Census]
        model.run()

        assert np.any(model.people.S[-1] < model.people.S[0]), "Susceptible: deaths not occurring."
        assert np.any(model.people.E[-1] < model.people.E[0]), "Exposed: deaths not occurring."
        assert np.any(model.people.Isym[-1] < model.people.Isym[0]), "Infectious (symptomatic): deaths not occurring."
        assert np.any(model.people.Iasym[-1] < model.people.Iasym[0]), "Infectious (symptomatic): deaths not occurring."
        assert np.any(model.people.R[-1] < model.people.R[0]), "Recovered: deaths not occurring."
        assert np.any(model.people.V1[-1] < model.people.V1[0]), "Vaccinated (one dose, immune): deaths not occurring."
        assert np.any(model.people.V2[-1] < model.people.V2[0]), "Vaccinated (two doses, immune): deaths not occurring."

        # assert np.any(model.people.V2inf[-1] < model.people.V2inf[0]), "Vaccinated (two doses, infected): deaths not occurring."

        assert np.all(model.people.S[-1] <= model.people.S[0]), "Susceptible: unexpected population increase."
        assert np.all(model.people.E[-1] <= model.people.E[0]), "Exposed: unexpected population increase."
        assert np.all(model.people.Isym[-1] <= model.people.Isym[0]), "Infectious (symptomatic): unexpected population increase."
        assert np.all(model.people.Iasym[-1] <= model.people.Iasym[0]), "Infectious (symptomatic): unexpected population increase."
        assert np.all(model.people.R[-1] <= model.people.R[0]), "Recovered: unexpected population increase."
        assert np.all(model.people.V1[-1] <= model.people.V1[0]), "Vaccinated (one dose, immune): unexpected population increase."
        assert np.all(model.people.V2[-1] <= model.people.V2[0]), "Vaccinated (two doses, immune): unexpected population increase."

        # assert np.all(model.people.V2inf[-1] <= model.people.V2inf[0]), "Vaccinated (two doses, infected): unexpected population increase."
        aggregate = model.people.S + model.people.E + model.people.Isym + model.people.Iasym + model.people.R + model.people.V1 + model.people.V2
        assert np.all(model.patches.non_disease_deaths[:-1] == (aggregate[:-1] - aggregate[1:])), "total population: deaths not recorded correctly."

        return


if __name__ == "__main__":
    unittest.main()
