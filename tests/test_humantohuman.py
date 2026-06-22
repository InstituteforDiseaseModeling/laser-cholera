import unittest
from datetime import datetime

import numpy as np

from laser.cholera.metapop.census import Census
from laser.cholera.metapop.exposed import Exposed
from laser.cholera.metapop.humantohuman import HumanToHuman
from laser.cholera.metapop.infectious import Infectious
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.recovered import Recovered
from laser.cholera.metapop.susceptible import Susceptible
from laser.cholera.utils import sim_duration


class TestHumanToHuman(unittest.TestCase):
    @staticmethod
    def get_test_parameters():
        params = get_parameters(mods=sim_duration(datetime(2024, 1, 1), datetime(2024, 12, 31)), do_validation=False)
        iNigeria = params.location_name.index("NGA")
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        params.I_j_initial[iNigeria] = 100
        params.S_j_initial[iNigeria] -= 100

        return params, iNigeria

    # What if tau_i = 0? (no emmigration)
    def test_humantohuman_tau_i_zero(self):
        params, iNigeria = self.get_test_parameters()

        # Turn off emmigration
        params.tau_i[:] = 0

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model.run()

        for index in range(len(params.location_name)):
            if index != iNigeria:
                # No infected individuals outside Nigeria and no migration, expect Lambda to be zero
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
            else:
                # Infected individuals in Nigeria, expect Lambda to be non-zero
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."

        return

    # What if tau_i = 1? (total emmigration, no local transmission)
    def test_humantohuman_tau_i_one(self):
        params, iNigeria = self.get_test_parameters()

        # Setup no immigration into Nigeria but no local transmission due to complete emmigration
        params.tau_i[iNigeria] = 1.0

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        # Have to do this _after_ setting model components because HumanToHuman builds the pi_ij matrix
        model.patches.pi_ij[:, iNigeria] = 0.0

        model.run()

        # With no immigration and no local transmission, expect Lambda to be zero
        assert np.all(model.patches.Lambda[:, iNigeria] == 0), "HumanToHuman: tau_i = 1, Lambda_jt not zero in Nigeria."

        return

    # Special cases for pi_ij? (no connectivity)
    def test_humantohuman_pi_ij_zero(self):
        params, iNigeria = self.get_test_parameters()

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]

        # Turn off all migration, have to do this _after_ setting model components because HumanToHuman builds the pi_ij matrix
        model.patches.pi_ij *= 0.0

        model.run()

        for index in range(len(params.location_name)):
            if index != iNigeria:
                # With no infected people and no migration, expect Lambda to be zero outside Nigeria
                assert np.all(model.patches.Lambda[:, index] == 0), "HumanToHuman: tau_i = 0, Lambda_jt not zero."
            else:
                # With infected people in Nigeria, expect Lambda to be non-zero
                assert np.any(model.patches.Lambda[:, index] != 0), "HumanToHuman: tau_i = 0, Lambda_jt zero in Nigeria."

        return

    # Special cases for beta_j0_hum?
    # Special cases for beta_jt_human?

    def test_humantohuman_per_patch_alpha_matches_scalar(self):
        """A uniform `(npatches,)` `alpha_*` array produces identical state to the matching scalar.

        Given two parameter sets seeded identically — one with scalar
        `alpha_1` / `alpha_2`, the other with `np.full(npatches, scalar)`
        arrays of the same values — when both run through the same HumanToHuman
        + companion pipeline, then `model.patches.Lambda`, `model.people.S`,
        `model.people.E`, and `model.patches.incidence_human` are bit-equal
        across the two runs.

        This pins the broadcast invariance: `np.power(arr, scalar)` must equal
        `np.power(arr, np.full_like(arr, scalar))` element-wise, and every
        downstream stochastic draw (which depends on the seeded PRNG and the
        rate values) must therefore produce identical samples.

        Failure implies a regression where the dual-mode alpha path is not
        broadcast-equivalent to the scalar path — either the ingestion produced
        a subtly different dtype, or `np.power` is behaving differently between
        the two shapes, or the consumer is reading `alpha_*` in a way that
        treats the two forms non-equivalently.
        """
        params_scalar, _ = self.get_test_parameters()
        params_array, _ = self.get_test_parameters()

        # Pin both to the same in-range values; ingestion will have left these
        # as the bundled defaults (which may already be scalar).
        npatches = len(params_scalar.location_name)
        scalar_alpha_1 = np.float32(0.95)
        scalar_alpha_2 = np.float32(1.0)
        params_scalar.alpha_1 = scalar_alpha_1
        params_scalar.alpha_2 = scalar_alpha_2
        params_array.alpha_1 = np.full(npatches, scalar_alpha_1, dtype=np.float32)
        params_array.alpha_2 = np.full(npatches, scalar_alpha_2, dtype=np.float32)

        model_scalar = Model(params_scalar)
        model_scalar.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model_scalar.run()

        model_array = Model(params_array)
        model_array.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model_array.run()

        np.testing.assert_array_equal(
            model_scalar.patches.Lambda,
            model_array.patches.Lambda,
            err_msg="Lambda differs between scalar-alpha and uniform-array-alpha runs — broadcast invariance broken",
        )
        np.testing.assert_array_equal(
            model_scalar.people.S,
            model_array.people.S,
            err_msg="S compartment differs between scalar-alpha and uniform-array-alpha runs",
        )
        np.testing.assert_array_equal(
            model_scalar.people.E,
            model_array.people.E,
            err_msg="E compartment differs between scalar-alpha and uniform-array-alpha runs",
        )
        np.testing.assert_array_equal(
            model_scalar.patches.incidence_human,
            model_array.patches.incidence_human,
            err_msg="incidence_human differs between scalar-alpha and uniform-array-alpha runs",
        )

        return

    def test_humantohuman_per_patch_alpha_indexes_correctly(self):
        """A per-patch perturbation of `alpha_1[j]` affects only patch `j`'s Lambda at the first tick.

        Given two parameter sets with mobility off (`tau_i = 0`) and identical
        initial state, but `alpha_1` arrays that differ at exactly one patch
        index `j` (and nowhere else), when each model runs, then at the first
        non-zero Lambda step (`patches.Lambda[1, :]`, which is deterministic —
        a function of the shared initial state and parameters, no stochastic
        draws yet):

        - `Lambda[1, k]` is bit-equal between the two runs for every patch
          `k != j`.
        - `Lambda[1, j]` differs between the two runs.

        This pins the load-bearing "alpha[j] is applied to patch j" claim of
        the per-patch acceptance criterion. With mobility off the patches are
        decoupled in the FOI computation, so a perturbation at patch `j`
        cannot leak into another patch's Lambda via the spatial coupling —
        any leakage would be a per-patch-indexing bug.

        Failure implies the consumer (`humantohuman.py`'s `np.power`) is
        reading `alpha_1` along the wrong axis or broadcasting incorrectly,
        so the per-patch exponents are being applied to the wrong patches.
        """
        params_a, _ = self.get_test_parameters()
        params_b, _ = self.get_test_parameters()

        # Mobility off — patches decouple in the FOI; isolating the
        # per-patch indexing claim from any cross-patch spatial spillover.
        params_a.tau_i[:] = 0.0
        params_b.tau_i[:] = 0.0

        npatches = len(params_a.location_name)
        # Pick a patch that has non-zero infectious people so `effective_i[j] > 0`
        # and `np.power(effective_i, alpha_1[j])` actually depends on the exponent.
        j = int(np.argmax(params_a.I_j_initial))

        # Use uniform arrays so the scalar-vs-array path differences cancel,
        # then perturb only one patch's alpha_1 between the two runs.
        params_a.alpha_1 = np.full(npatches, 0.95, dtype=np.float32)
        params_b.alpha_1 = np.full(npatches, 0.95, dtype=np.float32)
        params_b.alpha_1[j] = np.float32(0.85)  # only this entry differs

        model_a = Model(params_a)
        model_a.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model_a.run()

        model_b = Model(params_b)
        model_b.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model_b.run()

        lambda_a = model_a.patches.Lambda[1, :]
        lambda_b = model_b.patches.Lambda[1, :]

        # Every patch other than `j` must have bit-equal Lambda at the first
        # FOI step (deterministic from initial state + parameters).
        for k in range(npatches):
            if k == j:
                continue
            assert lambda_a[k] == lambda_b[k], (
                f"Lambda[1, {k}] differs between runs that differ only in alpha_1[{j}] "
                f"({lambda_a[k]} vs {lambda_b[k]}) — alpha indexing may be leaking across patches"
            )

        # And the perturbed patch's Lambda MUST differ, otherwise the
        # perturbation was inert and we have no proof it was applied at all.
        assert lambda_a[j] != lambda_b[j], (
            f"Lambda[1, {j}] is bit-equal between alpha_1[{j}]=0.95 and alpha_1[{j}]=0.85 runs — the per-patch exponent was not applied"
        )

        return

    def test_humantohuman_per_patch_alpha_runs_end_to_end(self):
        """A heterogeneous per-patch `alpha_1` / `alpha_2` produces a finite, non-trivial Lambda.

        Given a parameter set whose `alpha_1` and `alpha_2` are length-`npatches`
        arrays with deliberately heterogeneous in-range values (each patch gets
        a distinct exponent), when the model runs end-to-end, then
        `model.patches.Lambda` contains no NaN / inf entries and has at least
        one non-zero entry — i.e., the FOI machinery ran cleanly and produced
        signal somewhere.

        Failure implies the heterogeneous-array path produces NaN / inf
        (e.g., from `np.power` with a pathological exponent), or the FOI
        collapses to zero everywhere despite an active outbreak in Nigeria.
        """
        params, _ = self.get_test_parameters()
        npatches = len(params.location_name)

        # Heterogeneous in-range exponents: alpha_1 sweeps (0, 1], alpha_2 sweeps [0, 1].
        params.alpha_1 = np.linspace(0.5, 1.0, npatches, dtype=np.float32)
        params.alpha_2 = np.linspace(0.0, 1.0, npatches, dtype=np.float32)

        model = Model(params)
        model.components = [Susceptible, Exposed, Infectious, Recovered, Census, HumanToHuman]
        model.run()

        assert np.all(np.isfinite(model.patches.Lambda)), "HumanToHuman produced NaN / inf Lambda with heterogeneous alpha arrays."
        assert np.any(model.patches.Lambda != 0.0), (
            "HumanToHuman produced all-zero Lambda — the heterogeneous-alpha path may be silently zeroing the FOI."
        )

        return


if __name__ == "__main__":
    unittest.main()
