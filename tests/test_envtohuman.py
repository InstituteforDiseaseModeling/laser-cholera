import unittest

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.environmental import Environmental
from laser_cholera.metapop.envtohuman import EnvToHuman
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.test import Eradication
from laser_cholera.utils import sim_duration


class TestEnvToHuman(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = cls.get_test_parameters()
        cls.baseline = Model(cls.params)
        cls.baseline.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        cls.baseline.run()

        return

    @staticmethod
    def get_test_parameters():
        params = get_parameters(overrides=sim_duration(), do_validation=False)
        params.S_j_initial += params.I_j_initial  # return initial I to S  # ty:ignore[unresolved-attribute]
        params.I_j_initial = 50_000  # fix I at 50,000 # ty:ignore[unresolved-attribute]
        params.S_j_initial -= params.I_j_initial  # remove I from S # ty:ignore[unresolved-attribute]

        return params

    # Test no WASH interventions
    def test_envtohuman_no_wash(self):
        params = self.get_test_parameters()

        # Zero out WASH coverage
        params.theta_j *= 0.0

        model = Model(params)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        model.run()

        # At t = 0, Environmental component seeds environmental reservoir at t = 1
        # At t = 1, EnvToHuman components calculates newly exposed people at t = 2

        # Expect exposed at t = 2 to be _more_ for the test model.
        assert np.all(model.people.E[2] >= self.baseline.people.E[2]), "EnvToHuman: exposed not increasing with no WASH coverage."

        return

    # Test perfect WASH interventions
    def test_envtohuman_perfect_wash(self):
        params = self.get_test_parameters()

        # Perfect WASH coverage
        params.theta_j = 1.0

        model = Model(params)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        model.run()

        # At t = 0, Environmental component seeds environmental reservoir at t = 1
        # At t = 1, EnvToHuman components calculates newly exposed people at t = 2

        # Expect exposed at t = 2 to be 0 with perfect WASH intervention.
        assert np.all(model.people.E[2] == 0), "EnvToHuman: exposed not 0 with perfect WASH coverage."

        return

    # Test increased seasonal factors
    def test_envtohuman_increased_seasonality(self):
        params = self.get_test_parameters()

        # Increase seasonal factors
        params.beta_j0_env *= 5.0

        model = Model(params)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        model.run()

        # Expect exposed at t = 2 to be _more_ for the test model.
        assert np.all(model.people.E[2] >= self.baseline.people.E[2]), "EnvToHuman: exposed not increasing with increased seasonal factors."

        return

    # Test decreased seasonal factors
    def test_envtohuman_decreased_seasonality(self):
        params = self.get_test_parameters()

        # Decrease seasonal factors
        params.beta_j0_env /= 5.0

        model = Model(params)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        model.run()

        # Expect exposed at t = 2 to be _less_ for the test model.
        assert np.all(model.people.E[2] <= self.baseline.people.E[2]), "EnvToHuman: exposed not decreasing with decreased seasonal factors."

        return

    # Test lower infection threshold
    def test_envtohuman_lower_infection_threshold(self):
        params = self.get_test_parameters()

        # Decrease infection threshold
        params.kappa /= 5.0

        model = Model(params)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        model.run()

        # Expect exposed at t = 2 to be _more_ for the test model.
        assert np.all(model.people.E[2] >= self.baseline.people.E[2]), "EnvToHuman: exposed not increasing with lower infection threshold."

        return

    # Test higher infection threshold
    def test_envtohuman_higher_infection_threshold(self):
        params = self.get_test_parameters()

        # Increase infection threshold
        params.kappa *= 5.0

        model = Model(params)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, EnvToHuman, Census, Environmental]
        model.run()

        # Expect exposed at t = 2 to be _less_ for the test model.
        assert np.all(model.people.E[2] <= self.baseline.people.E[2]), "EnvToHuman: exposed not decreasing with higher infection threshold."

        return


if __name__ == "__main__":
    unittest.main()
