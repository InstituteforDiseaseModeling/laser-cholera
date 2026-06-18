import unittest
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pytest

from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.utils import UnknownOverrideKey
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

    def test_new_scalar_coercions(self):
        """Recently added scalar entries (`sigma`, `rho_deaths`, etc.) coerce to numbers.

        Given a battery of new scalar keys added when `override_helper` was
        reconciled with `default_parameters.json` (``sigma``, ``rho_deaths``,
        ``chi_endemic``, ``chi_epidemic``, ``zeta_ratio``,
        ``delta_reporting_cases``, ``delta_reporting_deaths``,
        ``decay_days_spread``),
        when each is passed in as a stringly-typed override,
        then the coerced value is the expected numeric type with the
        expected magnitude.

        Failure implies the JSON-vs-mapping reconciliation has regressed
        and a CLI ``--over sigma:0.5`` style call would either silently
        forward the string or raise an unexpected error.
        """
        typed = override_helper(
            {
                "sigma": "0.5",
                "rho_deaths": "0.1",
                "chi_endemic": "0.2",
                "chi_epidemic": "0.3",
                "zeta_ratio": "1.5",
                "delta_reporting_cases": "7",
                "delta_reporting_deaths": "14",
                "decay_days_spread": "30",
            }
        )
        assert isinstance(typed["sigma"], float)
        assert typed["sigma"] == 0.5
        assert isinstance(typed["rho_deaths"], float)
        assert typed["rho_deaths"] == 0.1
        assert isinstance(typed["chi_endemic"], float)
        assert typed["chi_endemic"] == 0.2
        assert isinstance(typed["chi_epidemic"], float)
        assert typed["chi_epidemic"] == 0.3
        assert isinstance(typed["zeta_ratio"], float)
        assert typed["zeta_ratio"] == 1.5
        assert isinstance(typed["delta_reporting_cases"], int)
        assert typed["delta_reporting_cases"] == 7
        assert isinstance(typed["delta_reporting_deaths"], int)
        assert typed["delta_reporting_deaths"] == 14
        assert isinstance(typed["decay_days_spread"], int)
        assert typed["decay_days_spread"] == 30

    def test_unknown_key_raises_with_difflib_suggestion(self):
        """A misspelled override key raises `UnknownOverrideKey` with a hint.

        Given an overrides dict whose key (``date_strat``) is one character
        off from a real mapping entry (``date_start``),
        when ``override_helper`` runs,
        then it raises `UnknownOverrideKey` (a `ValueError` subclass) whose
        message names the bad key and offers a `Did you mean '…'?`
        suggestion produced by `difflib.get_close_matches`.

        Failure implies the strict-key check or the difflib suggestion
        path has regressed; misspelled CLI flags would silently no-op
        again and surface later as confusing shape errors.
        """
        with pytest.raises(UnknownOverrideKey) as ctx:
            override_helper({"date_strat": "2024-01-01"})
        assert "date_strat" in str(ctx.exception)
        assert "date_start" in str(ctx.exception)
        # Subclass relationship is part of the contract — callers may
        # catch the broader ValueError and still match.
        assert isinstance(ctx.exception, ValueError)

    def test_unknown_key_with_no_close_match_omits_suggestion(self):
        """Unknown keys with no close match produce a clean, suggestion-free message.

        Given an overrides dict whose key (``xyz_garbage_zzz``) is too far
        from any mapping entry to clear the difflib similarity cutoff,
        when ``override_helper`` runs,
        then the raised `UnknownOverrideKey` names the bad key but omits
        any `Did you mean` suffix.

        Failure implies the difflib cutoff is producing nonsense
        suggestions or the suggestion-suffix branch is wired wrong.
        """
        with pytest.raises(UnknownOverrideKey) as ctx:
            override_helper({"xyz_garbage_zzz": "value"})
        assert "xyz_garbage_zzz" in str(ctx.exception)
        assert "Did you mean" not in str(ctx.exception)

    def test_cli_unsupported_keys_reject_with_helpful_message(self):
        """Vector / matrix / DataFrame parameter overrides are rejected at the CLI boundary.

        Given an overrides dict containing keys that are valid model
        parameters but whose values must be non-scalar (`S_j_initial` —
        vector, `b_jt` — matrix, `epidemic_peaks` — DataFrame, `return` —
        list),
        when ``override_helper`` runs,
        then each raises a plain `ValueError` (NOT `UnknownOverrideKey`)
        whose message names the rejected key and points at `--params` /
        `get_parameters` as the escape hatches.

        Failure implies the `_cli_unsupported` factory regressed; the
        previous behavior of silently forwarding a string to a position
        expecting a 40-long vector would corrupt the simulation later
        with a confusing shape mismatch.
        """
        unsupported = ["S_j_initial", "b_jt", "epidemic_peaks", "return", "psi_jt", "nu_jt_sources"]
        for key in unsupported:
            with pytest.raises(ValueError) as ctx:
                override_helper({key: "anything"})
            # Must NOT be UnknownOverrideKey — these keys ARE known.
            assert not isinstance(ctx.exception, UnknownOverrideKey), f"{key} should reject as plain ValueError, not UnknownOverrideKey"
            message = str(ctx.exception)
            assert key in message, f"{key} not mentioned in error message"
            assert "--params" in message, f"{key} message should suggest --params"


if __name__ == "__main__":
    unittest.main()
