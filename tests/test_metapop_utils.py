import unittest

import numpy as np

from laser_cholera.metapop.params import get_parameters
from laser_cholera.metapop.utils import get_daily_seasonality
from laser_cholera.metapop.utils import get_pi_from_lat_long


class TestMetapopUtils(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = get_parameters()
        cls.npatches = len(cls.params.location_name)

        return

    def test_get_daily_seasonality(self):
        seasonality = get_daily_seasonality(self.params)
        assert seasonality.shape == (self.params.nticks, self.npatches), "get_daily_seasonality: seasonality shape mismatch"
        assert seasonality.dtype == np.float32, "get_daily_seasonality: seasonality dtype mismatch"

        # TODO - test some values?

        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(16, 9), dpi=128, num="Seasonality")
        beta_j0 = params.beta_j0_hum
        a1 = params.a_1_j
        b1 = params.b_1_j
        a2 = params.a_2_j
        b2 = params.b_2_j
        p = params.p
        t = np.arange(0, p)
        for i in range(npatches):
            ax = plt.subplot(8, 5, i + 1)
            cos1 = a1[i] * np.cos(2 * np.pi * t / p)
            sin1 = b1[i] * np.sin(2 * np.pi * t / p)
            cos2 = a2[i] * np.cos(4 * np.pi * t / p)
            sin2 = b2[i] * np.sin(4 * np.pi * t / p)
            plt.plot(beta_j0[i] * (1.0 + cos1 + sin1 + cos2 + sin2))
            ax.set_title(f"{params.location_name[i]}")
        # plt.tight_layout()
        plt.show()
        """

        return

    def test_get_pi_from_lat_long(self):
        pi_ij = get_pi_from_lat_long(self.params)
        assert pi_ij.shape == (self.npatches, self.npatches), "get_pi_from_lat_long: pi_ij shape mismatch"
        assert pi_ij.dtype == np.float32, "get_pi_from_lat_long: pi_ij dtype mismatch"
        assert np.all(pi_ij.diagonal() == 0), "get_pi_from_lat_long: pi_ij diagonal should be zero"
        assert np.all(pi_ij >= 0), "get_pi_from_lat_long: pi_ij should be non-negative"
        assert np.all(pi_ij.sum(axis=1) <= 1.000001), "get_pi_from_lat_long: pi_ij should not specify more than 100% of the population"

        # TODO - test some values?

        return


if __name__ == "__main__":
    unittest.main()
