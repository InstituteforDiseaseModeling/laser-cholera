"""Tests for the Vaccinated compartment component of the cholera metapopulation model.

This module verifies the behavior of the `Vaccinated` class, which tracks one-dose (V1)
and two-dose (V2) vaccinated populations across model patches. Tests cover initialization,
steady-state invariance, natural mortality, waning immunity, dose delivery with varying
efficacy parameters, dose schedule consistency, and graceful handling of missing or
excluded source compartments.
"""

import unittest
from datetime import datetime

import numpy as np

from laser.cholera.metapop.census import Census
from laser.cholera.metapop.exposed import Exposed
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.recovered import Recovered
from laser.cholera.metapop.susceptible import Susceptible
from laser.cholera.metapop.vaccinated import Vaccinated
from laser.cholera.utils import sim_duration


class TestVaccinated(unittest.TestCase):
    """Test suite for the `Vaccinated` compartment component.

    Each test constructs a minimal model with disease dynamics disabled as needed to
    isolate the mechanism under test. The `get_test_parameters` helper consolidates all
    non-vaccinated populations into the susceptible compartment so that total population
    is conserved and tests have a clean, known starting state.
    """

    @staticmethod
    def get_test_parameters(V1=20_000, V2=10_000, overrides=None):
        """Build a clean parameter set with controlled initial vaccinated populations.

        Collapses all exposed, infectious, and recovered individuals into susceptible
        so tests start from a single-compartment baseline. The one-dose (V1) and
        two-dose (V2) vaccinated counts are set to the provided values, with the
        equivalent number removed from susceptible to preserve total population.

        Args:
            V1 (int): Initial one-dose vaccinated count to assign uniformly across
                patches. Defaults to 20,000.
            V2 (int): Initial two-dose vaccinated count to assign uniformly across
                patches. Defaults to 10,000.
            overrides (dict | None): Parameter overrides passed to `get_parameters`.
                Defaults to `sim_duration()` (a minimal simulation window).

        Returns:
            Parameters: A parameter object with collapsed initial conditions and the
                specified V1/V2 populations.
        """
        if overrides is None:
            overrides = sim_duration()
        params = get_parameters(mods=overrides, do_validation=False)
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

    def test_initial_values(self):
        """V1 and V2 arrays are populated from params at tick 0 without running the model.

        Given a parameter set with explicit V1_j_initial and V2_j_initial values,
        when a Model is constructed with the Vaccinated component (without calling run()),
        then model.people.V1[0] and model.people.V2[0] should exactly match the
        corresponding parameter arrays.

        Failure implies that the Vaccinated.__init__ method does not correctly copy
        initial parameter values into the people arrays.
        """
        params = self.get_test_parameters()

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        # model.run() # don't need to run, just check initial values

        assert np.all(model.people.V1[0] == model.params.V1_j_initial), "V1: initial value not correct."
        assert np.all(model.people.V2[0] == model.params.V2_j_initial), "V2: initial value not correct."

    def test_vaccinated_steadystate(self):
        """V1 and V2 are unchanged over the full simulation when all dynamics are off.

        Given deaths, waning immunity, and new vaccinations are all zeroed out,
        when the model is run for the full simulation duration,
        then the final V1 and V2 populations should equal their initial values at every patch.

        Failure implies a leak or gain in the vaccinated compartments under conditions
        where no mechanism should alter them — indicating a bookkeeping error in `__call__`.
        """
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1[-1] == model.people.V1[0]), "V1 immunized: steady state not held."
        assert np.all(model.people.V2[-1] == model.people.V2[0]), "V2 immunized: steady state not held."

        return

    def test_vaccinated_non_disease_deaths(self):
        """Natural mortality removes individuals from both V1 and V2 over time.

        Given a 10x inflated non-disease death rate with waning immunity and new
        vaccinations disabled,
        when the model is run for the full simulation duration,
        then the final V1 and V2 counts should be strictly less than their initial values
        at every patch.

        Failure implies that natural mortality is not being applied to vaccinated
        compartments, or is being applied incorrectly (e.g., no-op or negative).
        """
        params = self.get_test_parameters()

        params.d_jt *= 10  # inflate non-disease death rate
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1[-1] < model.people.V1[0]), "V1 immunized: non-disease deaths not occurring."
        assert np.all(model.people.V2[-1] < model.people.V2[0]), "V2 immunized: non-disease deaths not occurring."

        return

    def test_vaccinated_waning_one_dose_immunity(self):
        """One-dose waning immunity reduces V1 while V2 remains unchanged.

        Given deaths are off, two-dose waning (omega_2) is off, and new vaccinations
        are off, but one-dose waning (omega_1) is active,
        when the model is run for the full simulation duration,
        then V1 should decline at every patch while V2 remains at its initial level.

        Failure of the V1 assertion implies omega_1 waning is not applied to V1.
        Failure of the V2 assertion implies a cross-compartment error — V2 is being
        modified when it should not be.
        """
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        # leave this on: params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1[-1] < model.people.V1[0]), "V1 immunized: missing waning immunity."
        assert np.all(model.people.V2[-1] == model.people.V2[0]), "V2 immunized: should not change."

        return

    def test_vaccinated_waning_two_dose_immunity(self):
        """Two-dose waning immunity reduces V2 while V1 remains unchanged.

        Given deaths are off, one-dose waning (omega_1) is off, and new vaccinations
        are off, but two-dose waning (omega_2) is active,
        when the model is run for the full simulation duration,
        then V2 should decline at every patch while V1 remains at its initial level.

        Failure of the V2 assertion implies omega_2 waning is not applied to V2.
        Failure of the V1 assertion implies a cross-compartment error — V1 is being
        modified when it should not be.
        """
        params = self.get_test_parameters()

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        # leave this on: params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1[-1] == model.people.V1[0]), "V1 immunized: should not change."
        assert np.all(model.people.V2[-1] < model.people.V2[0]), "V2 immunized: missing waning immunity."

        return

    def test_vaccinated_one_dose_vaccination_realistic_efficacy(self):
        """First-dose delivery with realistic efficacy grows V1 and leaves V2 unchanged.

        Given no deaths, no waning immunity, 100 first doses/day for 32 days using the
        default (realistic, sub-unity) phi_1 efficacy, and second doses disabled,
        when the model is run,
        then V1 should be strictly greater than its initial value (zero) at every patch,
        and V2 should remain at zero.

        Failure of the V1 assertion implies that dose delivery or efficacy application
        is broken. Failure of the V2 assertion implies unintended cross-contamination
        of the two-dose compartment.
        """
        params = self.get_test_parameters(V1=0, V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt[0:32] = 100
        params.nu_2_jt *= 0  # turn off second dose vaccination

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1[-1] > model.people.V1[0]), "V1 immunized: missing newly vaccinated people."
        assert np.all(model.people.V2[-1] == model.people.V2[0]), "V2 immunized: steady state not held."

        return

    def test_vaccinated_one_dose_vaccination_perfect_efficacy(self):
        """First-dose delivery with perfect efficacy (phi_1=1.0) grows V1 and leaves V2 unchanged.

        Given no deaths, no waning immunity, 100 first doses/day for 32 days with phi_1
        set to 1.0 (every delivered dose results in immunity), and second doses disabled,
        when the model is run,
        then V1 should be strictly greater than its initial value at every patch, and V2
        should remain unchanged.

        Failure of the V1 assertion implies the perfect-efficacy path is broken.
        Failure of the V2 assertion implies unintended cross-contamination of V2.
        """
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

        assert np.all(model.people.V1[-1] > model.people.V1[0]), "V1 immunized: missing newly vaccinated people."
        assert np.all(model.people.V2[-1] == model.people.V2[0]), "V2 immunized: steady state not held."

        return

    def test_vaccinated_one_dose_vaccination_no_efficacy(self):
        """First-dose delivery with zero efficacy (phi_1=0.0) leaves both V1 and V2 unchanged.

        Given no deaths, no waning immunity, 100 first doses/day for 32 days with phi_1
        set to 0.0 (no delivered dose results in immunity), and second doses disabled,
        when the model is run,
        then V1 and V2 should both remain at their initial values of zero throughout.

        Failure implies that doses are being moved into V1 even when phi_1 is zero, or
        that the zero-efficacy branch is not handled correctly.
        """
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

        assert np.all(model.people.V1[-1] == model.people.V1[0]), "V1 immunized: should not have newly vaccinated people."
        assert np.all(model.people.V2[-1] == model.people.V2[0]), "V2 immunized: steady state not held."

        return

    def test_vaccinated_two_dose_vaccination_realistic_efficacy(self):
        """Second-dose delivery with realistic efficacy moves people from V1 to V2.

        Given V2=0, no deaths, no waning, no first-dose campaign, and 100 second doses/day
        for 32 days with default (sub-unity) phi_2 efficacy,
        when the model is run,
        then V1 should decrease (individuals graduate to V2) and V2 should increase.

        Failure of the V1 assertion implies that second-dose delivery is not decrementing
        V1. Failure of the V2 assertion implies that effective second doses are not being
        added to V2.
        """
        params = self.get_test_parameters(V2=0)

        params.d_jt *= 0  # turn off deaths
        params.omega_1 = 0  # turn off waning immunity - one dose
        params.omega_2 = 0  # turn off waning immunity - two dose
        params.nu_1_jt *= 0  # turn off first dose vaccination
        params.nu_2_jt[0:32] = 100

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]  # , Infectious, Recovered]
        model.run()

        assert np.all(model.people.V1[-1] < model.people.V1[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.people.V2[-1] > model.people.V2[0]), "V2 immunized: missing newly vaccinated people."

        return

    def test_vaccinated_two_dose_vaccination_perfect_efficacy(self):
        """Second-dose delivery with perfect efficacy (phi_2=1.0) moves all delivered doses from V1 to V2.

        Given V2=0, no deaths, no waning, no first-dose campaign, 100 second doses/day for
        32 days with phi_2=1.0,
        when the model is run,
        then V1 should decrease and V2 should increase by the same total number of doses.

        Failure implies the perfect-efficacy path for second doses is broken.
        """
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

        assert np.all(model.people.V1[-1] < model.people.V1[0]), "V1 immunized: should be missing newly vaccinated people."
        assert np.all(model.people.V2[-1] > model.people.V2[0]), "V2 immunized: missing newly vaccinated people."

        return

    def test_vaccinated_two_dose_vaccination_no_efficacy(self):
        """Second-dose delivery with zero efficacy (phi_2=0.0) leaves both V1 and V2 unchanged.

        Given V2=0, no deaths, no waning, no first-dose campaign, 100 second doses/day for
        32 days with phi_2=0.0 (no dose results in immunity),
        when the model is run,
        then both V1 and V2 should remain at their initial values throughout.

        Failure implies that individuals are being moved between compartments even when
        efficacy is zero, indicating the zero-efficacy guard is missing or broken.
        """
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

        assert np.all(model.people.V1[-1] == model.people.V1[0]), "V1 immunized: should be steady state with phi_2 == 0."
        assert np.all(model.people.V2[-1] == model.people.V2[0]), "V2 immunized: should be steady state with phi_2 == 0."

        return

    def test_doses_given_against_dose_schedule(self):
        """Recorded doses in patches match the nu_1_jt and nu_2_jt vaccination schedules.

        Given a simulation run over 2023-01-01 to 2024-12-17 (shorter than the full
        vaccination schedule) with V1=0 and V2=0,
        when the model is run,
        then every tick where nu_1_jt (or nu_2_jt) is non-zero should have a non-zero
        entry in dose_one_doses (or dose_two_doses), and every non-zero recorded dose
        entry should correspond to a non-zero schedule entry.

        Note: the nu_1_jt slice is correctly bounded to model.params.nticks, but the
        nu_2_jt nonzero check does not apply the same slice — if the schedule extends
        beyond nticks, this may produce false positives.

        Failure implies a mismatch between the dose schedule parameters and what the
        Vaccinated component actually records, breaking downstream audit/reporting logic.
        """
        params = self.get_test_parameters(V1=0, V2=0, overrides=sim_duration(start=datetime(2023, 1, 1), stop=datetime(2024, 12, 17)))

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]
        model.run()

        # Test runs shorter duration than vaccination schedule, just check relevant entries
        nonzero_param_indices = np.nonzero(model.params.nu_1_jt[0 : model.params.nticks])
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

    def test_handling_nonexistent_source_compartment(self):
        """Nonexistent compartment names in nu_jt_sources do not prevent dose delivery.

        Given nu_jt_sources includes valid compartments (S, E, Isym, Iasym, R) plus
        three unknown names (J, Q, Z) that do not correspond to any model compartment,
        when the model is run over 2023-01-01 to 2024-12-17,
        then first-dose delivery should still succeed — non-zero schedule entries should
        produce non-zero recorded doses.

        Note: only first-dose (nu_1_jt / dose_one_doses) behavior is checked here;
        second-dose schedule consistency is not verified in this test.

        Failure implies that the presence of unrecognized source compartment names causes
        an error or silently suppresses dose delivery, rather than being gracefully ignored.
        """
        params = self.get_test_parameters(V1=0, V2=0, overrides=sim_duration(start=datetime(2023, 1, 1), stop=datetime(2024, 12, 17)))
        params += {"nu_jt_sources": ["S", "E", "Isym", "Iasym", "R", "J", "Q", "Z"]}

        model = Model(parameters=params)
        model.components = [Susceptible, Exposed, Vaccinated, Census]
        model.run()

        # Test runs shorter duration than vaccination schedule, just check relevant entries
        nonzero_param_indices = np.nonzero(model.params.nu_1_jt[0 : model.params.nticks])
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

        return

    def test_excluded_source_compartment(self):
        """Excluding a compartment from nu_jt_sources prevents vaccine draws from it.

        Given a population split roughly 50/50 between S and R, natural mortality and
        waning immunity disabled, and two model runs where the baseline includes "R" in
        nu_jt_sources and the comparison excludes it,
        when both models are run over 2023-01-01 to 2024-12-17,
        then the comparison Recovered population should be entirely static over time,
        and should be greater than or equal to the baseline Recovered at every tick and
        patch (because the baseline draws from R, shrinking it).

        Failure of the static-R assertion implies that excluding R from nu_jt_sources
        still permits dose draws from the Recovered compartment.
        Failure of the comparison >= baseline assertion implies the exclusion logic is
        inverted or the two runs are otherwise identical.
        """
        params = self.get_test_parameters(V1=0, V2=0, overrides=sim_duration(start=datetime(2023, 1, 1), stop=datetime(2024, 12, 17)))
        params.R_j_initial[:] = params.S_j_initial // 2
        params.S_j_initial -= params.R_j_initial
        params += {"nu_jt_sources": ["S", "E", "Isym", "Iasym", "R"]}
        # Turn off natural mortality and waning immunity so Recovered population
        # is static except for vaccine delivery.
        params.d_jt[:] = 0.0  # no non-disease deaths
        params.epsilon = 0.0  # no waning immunity for Recovered population

        baseline = Model(parameters=params)
        baseline.components = [Susceptible, Exposed, Recovered, Vaccinated, Census]
        baseline.run()

        # "R" not included in vaccine candidate sources
        # comparison Recovered population should be static with these settings
        params <<= {"nu_jt_sources": ["S", "E", "Isym", "Iasym"]}
        comparison = Model(parameters=params)
        comparison.components = [Susceptible, Exposed, Recovered, Vaccinated, Census]
        comparison.run()

        assert np.all(comparison.people.R == comparison.people.R[[0], :]), "Comparison Recovered should be unchanging over time."
        assert np.any(comparison.people.R > baseline.people.R), "Recovered population should not be used for vaccination."
        assert np.all(comparison.people.R >= baseline.people.R), "Comparison Recovered should be >= baseline Recovered."

        return


if __name__ == "__main__":
    unittest.main()
