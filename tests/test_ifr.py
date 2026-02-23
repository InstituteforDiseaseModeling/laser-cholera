import unittest

import numpy as np

from laser.cholera.metapop import Census
from laser.cholera.metapop import Environmental
from laser.cholera.metapop import Exposed
from laser.cholera.metapop import Infectious
from laser.cholera.metapop import Recovered
from laser.cholera.metapop import Susceptible
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.utils import sim_duration


class TestIFRImplementation(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(mods=sim_duration(), do_validation=False)
        params.S_j_initial += params.I_j_initial  # return initial I to S
        params.I_j_initial = 10_000  # fix I at 10,000
        params.S_j_initial -= params.I_j_initial  # remove fixed, test I from S

        return params

    def test_higher_epidemic_threshold(self):
        ps = self.get_test_parameters()

        ps.epidemic_threshold = 0.0001
        ps.mu_j_epidemic_factor[:] = 0.5  # 50% higher during epidemic

        baseline = Model(ps)
        baseline.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        baseline.run()

        ps.epidemic_threshold *= 100  # 100x higher, effectively always in endemic mode, fewer deaths

        model = Model(ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        model.run()

        # disease deaths (model.patches.disease_deaths) should be lower
        # TODO - this is a low bar, can we make this more rigorous?
        assert model.patches.disease_deaths.sum() < baseline.patches.disease_deaths.sum(), (
            "Disease deaths should be lower with higher epidemic threshold."
        )

        return

    def test_lower_epidemic_threshold(self):
        ps = self.get_test_parameters()

        ps.epidemic_threshold = 0.0001
        ps.mu_j_epidemic_factor[:] = 0.5  # 50% higher during epidemic

        baseline = Model(ps)
        baseline.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        baseline.run()

        ps.epidemic_threshold /= 100  # 100x lower, more often in epidemic mode, more deaths

        model = Model(ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        model.run()

        # disease deaths (model.patches.disease_deaths) should be higher
        # TODO - this is a low bar, can we make this more rigorous?
        assert model.patches.disease_deaths.sum() > baseline.patches.disease_deaths.sum(), (
            "Disease deaths should be higher with lower epidemic threshold."
        )

        return

    def test_lower_epidemic_factor(self):
        ps = self.get_test_parameters()

        ps.epidemic_threshold = 0.0001
        ps.mu_j_epidemic_factor[:] = 0.5  # 50% higher during epidemic

        baseline = Model(ps)
        baseline.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        baseline.run()

        ps.mu_j_epidemic_factor /= 2  # 2x lower, smaller epidemic effect

        model = Model(ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        model.run()

        # disease deaths (model.patches.disease_deaths) should be lower
        # TODO - this is a low bar, can we make this more rigorous?
        assert model.patches.disease_deaths.sum() < baseline.patches.disease_deaths.sum(), (
            "Disease deaths should be lower with smaller epidemic effect."
        )

        return

    def test_higher_epidemic_factor(self):
        ps = self.get_test_parameters()

        ps.epidemic_threshold = 0.0001
        ps.mu_j_epidemic_factor[:] = 0.5  # 50% higher during epidemic

        baseline = Model(ps)
        baseline.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        baseline.run()

        ps.mu_j_epidemic_factor *= 1.5  # 50% higher, larger epidemic effect

        model = Model(ps)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        model.run()

        # disease deaths (model.patches.disease_deaths) should be higher
        # TODO - this is a low bar, can we make this more rigorous?
        assert model.patches.disease_deaths.sum() > baseline.patches.disease_deaths.sum(), (
            "Disease deaths should be higher with larger epidemic threshold."
        )

        return

    def test_zero_epidemic_factor(self):
        ps = self.get_test_parameters()

        ps.epidemic_threshold = 1.01
        ps.mu_j_epidemic_factor[:] = 0.5  # 50% higher during epidemic

        no_epidemic = Model(ps)
        no_epidemic.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        no_epidemic.run()

        ps.epidemic_threshold = 0.0001
        ps.mu_j_epidemic_factor[:] = 0  # no epidemic effect

        no_effect = Model(ps)
        no_effect.components = [Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        no_effect.run()

        # disease deaths (no_effect.patches.disease_deaths) should be same
        assert np.all(no_effect.patches.disease_deaths == no_epidemic.patches.disease_deaths), (
            "Disease deaths with zero epidemic factor should equal disease deaths with no epidemic."
        )

        return


if __name__ == "__main__":
    unittest.main()
