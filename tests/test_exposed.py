import unittest

import numpy as np

from laser.cholera.metapop.exposed import Exposed
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.utils import sim_duration


class TestExposed(unittest.TestCase):
    def test_exposed_steadystate(self):
        ps = get_parameters(mods=sim_duration(), do_validation=False)
        ps.d_jt *= 0  # turn off deaths
        model = Model(parameters=ps)
        model.components = [Exposed]
        model.run()
        assert np.all(model.people.E[-1] == model.people.E[0]), "Exposed: steady state not held."

        return

    def test_exposed_deaths(self):
        ps = get_parameters(mods=sim_duration(), do_validation=False)
        # Move some people to the exposed state so we have enough to see deaths
        ps.E_j_initial += 10_000
        ps.S_j_initial -= ps.E_j_initial
        model = Model(parameters=ps)
        model.components = [Exposed]
        model.run()
        assert np.all(model.people.E[-1] < model.people.E[0]), "Exposed: deaths not occurring."

        return


if __name__ == "__main__":
    unittest.main()
