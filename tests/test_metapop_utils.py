import unittest
from datetime import datetime
from types import SimpleNamespace

import numpy as np

from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.utils import get_daily_seasonality
from laser.cholera.metapop.utils import get_pi_from_lat_long
from laser.cholera.metapop.utils import override_helper


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

    def test_get_pi_from_lat_long_single_location(self):
        """A single-location parameter set yields a (1, 1) zero mobility matrix.

        Given a stub params object with one location (scalar latitude /
        longitude and length-1 population vectors),
        when ``get_pi_from_lat_long`` is called,
        then ``distance`` returns a 0-dim scalar that the function must
        promote to a 1×1 matrix, and the result is a 1×1 zero matrix because
        the diagonal is always 0 (no self-mobility).

        Failure implies the scalar-distance promotion branch (utils.py line
        ~54) has regressed and the function would crash on single-location
        configurations.
        """
        single = SimpleNamespace(
            latitude=np.float32(10.0),
            longitude=np.float32(20.0),
            mobility_omega=np.float32(1.0),
            mobility_gamma=np.float32(2.0),
            S_j_initial=np.array([1000], dtype=np.uint32),
            E_j_initial=np.array([0], dtype=np.uint32),
            I_j_initial=np.array([0], dtype=np.uint32),
            R_j_initial=np.array([0], dtype=np.uint32),
            V1_j_initial=np.array([0], dtype=np.uint32),
            V2_j_initial=np.array([0], dtype=np.uint32),
        )
        pi_ij = get_pi_from_lat_long(single)
        assert pi_ij.shape == (1, 1)
        # Diagonal is always 0; no other entries exist.
        assert pi_ij[0, 0] == 0.0


class TestOverrideHelper(unittest.TestCase):
    """Tests for ``override_helper`` — coerces stringly-typed CLI overrides.

    ``override_helper`` is used at the CLI boundary to type-coerce string
    values into the dtypes the rest of the pipeline expects. These tests pin
    the conversion table; failure usually means a CLI override silently
    forwards a string where a number was expected, which downstream throws as
    a confusing TypeError much later.
    """

    def test_scalar_numeric_overrides_are_typed(self):
        """Numeric scalar overrides are coerced to int/float per the mapping.

        Given an overrides dict containing ``phi_1`` (float-mapped),
        ``p`` (int-mapped), and ``iota`` (float-mapped),
        when ``override_helper`` runs,
        then each output value is the expected numeric type with the correct
        numerical content.

        Failure implies the mapping table or the coercion loop is broken
        and CLI numeric overrides would silently stay as strings.
        """
        typed = override_helper({"phi_1": "0.5", "p": "7", "iota": "1.25"})
        assert typed["phi_1"] == 0.5
        assert isinstance(typed["phi_1"], float)
        assert typed["p"] == 7
        assert isinstance(typed["p"], int)
        assert typed["iota"] == 1.25

    def test_date_overrides_are_parsed(self):
        """``date_start`` / ``date_stop`` overrides become ``datetime`` objects.

        Given ``date_start="2024-01-15"`` and ``date_stop="2024-12-31"``,
        when ``override_helper`` runs,
        then both values are ``datetime`` instances with the expected
        calendar components.

        Failure implies the partial-applied ``datetime.strptime`` either no
        longer receives the format kwarg or has been removed from the
        mapping table.
        """
        typed = override_helper({"date_start": "2024-01-15", "date_stop": "2024-12-31"})
        assert isinstance(typed["date_start"], datetime)
        assert typed["date_start"].year == 2024
        assert typed["date_start"].month == 1
        assert typed["date_start"].day == 15
        assert isinstance(typed["date_stop"], datetime)
        assert typed["date_stop"].year == 2024
        assert typed["date_stop"].month == 12
        assert typed["date_stop"].day == 31

    def test_bool_string_overrides_are_coerced(self):
        """Truthy/falsy strings are mapped to True/False via the bool-from-string helper.

        Given a battery of recognized truthy strings (``"true"``, ``"1"``,
        ``"yes"``, ``"y"``, ``"t"``, ``"on"``, ``"enabled"``) and falsy
        strings (``"false"``, ``"0"``, ``"no"``, ``"off"``),
        when each is passed in as a value for a bool-mapped key
        (``visualize``, ``pdf``, ``hdf5_output``, ``compress``, ``quiet``),
        then the truthy ones become ``True`` and falsy ones become ``False``.

        Failure implies the case-insensitive truthy-string set has shifted,
        which would break CLI ``--over visualize:on`` style invocations.
        """
        truthy_strings = ["true", "TRUE", "1", "yes", "y", "t", "on", "enabled"]
        falsy_strings = ["false", "0", "no", "off"]
        bool_keys = ["visualize", "pdf", "hdf5_output", "compress", "quiet"]

        for key in bool_keys:
            for s in truthy_strings:
                typed = override_helper({key: s})
                assert typed[key] is True, f"key={key} value={s} should map to True"
            for s in falsy_strings:
                typed = override_helper({key: s})
                assert typed[key] is False, f"key={key} value={s} should map to False"

    def test_unknown_keys_are_passed_through_unchanged(self):
        """Keys absent from the mapping are forwarded verbatim.

        Given an overrides dict with a key not in the type-coercion table
        (``some_new_param``),
        when ``override_helper`` runs,
        then the output contains the key with the original value type.

        Failure implies the function is dropping unknown keys silently,
        which would mask typos in CLI ``--over`` flags.
        """
        typed = override_helper({"some_new_param": "verbatim_value"})
        assert typed["some_new_param"] == "verbatim_value"

    def test_none_mapping_keys_pass_value_through_unchanged(self):
        """Keys mapped to ``None`` (vectors/matrices) keep their raw value.

        Given an overrides dict with a key whose mapping is ``None``
        (e.g., ``S_j_initial`` for population vectors, ``b_jt`` for a
        birth-rate matrix),
        when ``override_helper`` runs,
        then the output contains the key with the original value (no
        attempted coercion or wrapping).

        Failure implies the mapping table is now coercing vector-typed
        keys with the wrong function, which would corrupt array payloads
        sent in via the CLI ``--over`` mechanism.
        """
        vector_value = [1, 2, 3]
        typed = override_helper({"S_j_initial": vector_value, "b_jt": "matrix_stub"})
        assert typed["S_j_initial"] is vector_value
        assert typed["b_jt"] == "matrix_stub"


if __name__ == "__main__":
    unittest.main()
