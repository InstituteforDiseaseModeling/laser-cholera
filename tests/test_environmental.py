import unittest

import numpy as np

from laser_cholera.metapop.census import Census
from laser_cholera.metapop.environmental import Environmental
from laser_cholera.metapop.exposed import Exposed
from laser_cholera.metapop.infectious import Infectious
from laser_cholera.metapop.model import Model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.recovered import Recovered
from laser_cholera.metapop.susceptible import Susceptible
from laser_cholera.test import Eradication
from laser_cholera.utils import sim_duration


class TestEnvironmental(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(overrides=sim_duration(), do_validation=False)
        # Manipulate population
        params.S_j_initial += params.I_j_initial  # return initial I to S
        params.I_j_initial = 10_000  # fix I at 10,000
        params.S_j_initial -= params.I_j_initial  # remove I from S

        return params

    # Test steady state, delta_jt = 0, pulse of infected individuals
    def test_environmental_steadystate(self):
        ps = self.get_test_parameters()

        model = Model(ps)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, Census, Environmental]

        # No environmental decay
        model.patches.delta_jt *= 0.0

        model.run()

        # Expect environmental reservoir to be seeded at t = 1
        assert np.all(model.patches.W[1] > 1), "Environmental: reservoir not seeded."
        # Expect environmental reservoir to be constant over remaining time
        assert np.all(model.patches.W[2:] == model.patches.W[1]), "Environmental: reservoir not constant."

    # Test higher delta_jt, pulse of infected individuals, faster decay
    def test_environmental_high_delta(self):
        ps = self.get_test_parameters()

        # Baseline environmental parameters
        ps.decay_days_short = 3.0
        ps.decay_days_long = 90.0

        baseline = Model(ps)
        baseline.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        baseline.run()

        # Run with faster decay
        ps.decay_days_short = 2.0
        ps.decay_days_long = 45.0

        model = Model(ps)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        model.run()

        # Expect environmental reservoir to be seeded at t = 1 and same for both models
        assert np.all(baseline.patches.W[1] > 0), "Environmental: reservoir not seeded."
        assert np.all(model.patches.W[1] > 0), "Environmental: reservoir not seeded."
        assert np.all(model.patches.W[1] == baseline.patches.W[1]), "Environmental: reservoir not seeded equally."

        # Expect environmental contagion to be _less_ or _equal_ in the test model for t > 1
        # TODO - maybe change "<=" back to "<" after environmental factors are reviewed
        # However, it is okay if both baseline and test are zero.
        Wb = baseline.patches.W[2:]
        Wm = model.patches.W[2:]
        nz = np.nonzero(Wb)
        assert np.all(Wm[nz] <= Wb[nz]), "Environmental: reservoir not decaying faster."

        return

    # Test lower delta_jt, pulse of infected individuals, slower decay
    def test_environmental_low_delta(self):
        ps = self.get_test_parameters()

        # Baseline environmental parameters
        ps.decay_days_short = 3.0
        ps.decay_days_long = 90.0

        baseline = Model(ps)
        baseline.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        baseline.run()

        # Run with slower decay
        ps.decay_days_short = 6.0
        ps.decay_days_long = 120.0

        model = Model(ps)
        model.components = [Eradication, Susceptible, Exposed, Infectious, Recovered, Census, Environmental]
        model.run()

        # Expect environmental reservoir to be seeded at t = 1 and same for both models
        assert np.all(baseline.patches.W[1] > 0), "Environmental: reservoir not seeded."
        assert np.all(model.patches.W[1] > 0), "Environmental: reservoir not seeded."
        assert np.all(model.patches.W[1] == baseline.patches.W[1]), "Environmental: reservoir not seeded equally."

        # Expect environmental contagion to be _equal_ or _more_ in the test model for t > 1
        # TODO - maybe change ">=" back to ">" after environmental factors are reviewed
        assert np.all(model.patches.W[2:] >= baseline.patches.W[2:]), "Environmental: reservoir not decaying faster."

        return


if __name__ == "__main__":
    unittest.main()
