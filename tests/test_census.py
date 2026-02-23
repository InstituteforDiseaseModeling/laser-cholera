import unittest

import numpy as np

from laser.cholera.metapop.census import Census
from laser.cholera.metapop.exposed import Exposed
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.susceptible import Susceptible
from laser.cholera.utils import sim_duration


class TestCensus(unittest.TestCase):
    # Test with only susceptible people
    def test_census_susceptible(self):
        ps = get_parameters(mods=sim_duration(), do_validation=False)
        model = Model(parameters=ps)
        model.components = [Susceptible, Census]
        model.run()
        assert np.all(model.patches.N == model.people.S), "Census: total does not match susceptible."

        return

    # Test with only exposed people
    def test_census_exposed(self):
        ps = get_parameters(mods=sim_duration(), do_validation=False)
        model = Model(parameters=ps)
        ps.E_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model.components = [Exposed, Census]
        model.run()
        assert np.all(model.patches.N == model.people.E), "Census: total does not match exposed."

        return

    # Test with susceptible and exposed people
    def test_census_susceptible_exposed(self):
        ps = get_parameters(mods=sim_duration(), do_validation=False)
        model = Model(parameters=ps)
        ps.E_j_initial += 10_000
        ps.S_j_initial -= 10_000
        model.components = [Susceptible, Exposed, Census]
        model.run()
        assert np.all(model.patches.N == (model.people.S + model.people.E)), "Census: total does not match susceptible + exposed."

        return


if __name__ == "__main__":
    unittest.main()
