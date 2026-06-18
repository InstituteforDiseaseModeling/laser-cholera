"""Environmental-reservoir-to-human transmission component.

Computes a per-tick, per-patch environmental force-of-infection `Psi`
from the contaminated-water reservoir `patches.W` (maintained by
[`Environmental`][laser.cholera.metapop.environmental.Environmental])
modulated by a per-tick seasonal-suitability matrix `beta_jt_env`
derived from `params.psi_jt` and `params.beta_j0_env`. The
`(1 - theta_j)` factor models the fraction of the population NOT
covered by water/sanitation/hygiene (WASH), and the
`W / (kappa + W)` term saturates uptake as the reservoir grows.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key


class EnvToHuman:
    """Environmental transmission component: turns reservoir `W` into new `S -> E` infections.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate `Psi` and bake the `beta_jt_env` seasonality matrix.

        `beta_jt_env` is computed once at construction time from
        `params.psi_jt` (deviation-from-mean suitability) and
        `params.beta_j0_env` (baseline) and is constant across the run.

        Args:
            model: The `Model` instance. Must have `people`, `patches`,
                and `params` with `psi_jt`, `beta_j0_env` populated.

        Raises:
            AttributeError: When `model.people` or `model.params` is
                missing.
            ValueError: When `params.psi_jt` is missing.
            AssertionError: When the allocated `beta_jt_env` shape
                disagrees with the provided `psi_jt` / `beta_j0_env`
                shapes (these are invariant checks that remain).
        """
        self.model = model

        check_attr(model, "people", "EnvToHuman: model needs to have a 'people' attribute.")
        model.patches.add_vector_property("Psi", length=model.params.nticks + 1, dtype=np.float32, default=np.float32(0.0))

        check_attr(model, "params", "EnvToHuman: model needs to have a 'params' attribute.")
        check_key(model.params, "psi_jt", "EnvToHuman: model params needs to have a 'psi_jt' (environmental contamination rate) parameter.")

        psi = model.params.psi_jt  # convenience
        # TODO - use newer laser_core with add_array_property and psi.shape
        model.patches.add_vector_property("beta_jt_env", length=psi.shape[0], dtype=np.float32, default=0.0)
        assert model.patches.beta_jt_env.shape == model.params.psi_jt.shape
        assert model.params.beta_j0_env.shape[0] == model.patches.beta_jt_env.shape[1]
        psi_bar = psi.mean(axis=0, keepdims=True)
        model.patches.beta_jt_env[:, :] = model.params.beta_j0_env.T * (1.0 + (psi - psi_bar) / psi_bar)

        model.patches.add_vector_property("incidence_env", length=model.params.nticks + 1, dtype=np.int32, default=0)

        if not hasattr(model.patches, "incidence"):
            model.patches.add_vector_property("incidence", length=model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def check(self):
        """Validate the consumer-side state and transmission parameters.

        Asserts that `model.people.S` and `.E` exist, that
        `model.patches.W` exists (allocated by `Environmental`), and
        that `params` provides `tau_i`, `theta_j`, and `kappa`.

        Raises:
            AttributeError: When `model.people.S`, `.E`, `model.patches`,
                or `model.patches.W` is missing.
            ValueError: When `params.tau_i`, `theta_j`, or `kappa` is
                missing.
        """
        check_attr(self.model.people, "S", "EnvToHuman: model people needs to have a 'S' (susceptible) attribute.")
        check_attr(self.model.people, "E", "EnvToHuman: model people needs to have a 'E' (exposed) attribute.")

        check_attr(self.model, "patches", "EnvToHuman: model needs to have a 'patches' attribute.")
        check_attr(self.model.patches, "W", "EnvToHuman: model patches needs to have a 'W' (environmental) attribute.")

        check_key(self.model.params, "tau_i", "EnvToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter.")
        check_key(self.model.params, "theta_j", "EnvToHuman: model params needs to have a 'theta_j' (fraction of population with WASH) attribute.")
        check_key(self.model.params, "kappa", "EnvToHuman: model params needs to have a 'kappa' (environmental transmission rate) parameter.")

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Compute environmental force-of-infection `Psi` and convert `S -> E`.

        `Psi[tick + 1] = beta_jt_env[tick] * (1 - theta_j) * W /
        (kappa + W)`. New infections drawn from `Binomial(S_next,
        1 - exp(-Psi))`, where `S_next` is the susceptible count *after*
        natural mortality and any human-to-human force-of-infection has
        already been applied earlier in the tick.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.

        Raises:
            AssertionError: When the post-infection susceptible
                population goes negative.
        """
        Psi = model.patches.Psi[tick + 1]
        W = model.patches.W[tick]
        tau_i = model.params.tau_i
        local_frac = 1 - tau_i

        non_wash = (1 - model.params.theta_j) * W
        seasonal = model.patches.beta_jt_env[tick] * non_wash
        denominator = model.params.kappa + W
        normalized = seasonal / denominator
        Psi[:] = normalized

        # PsiS
        S_next = model.people.S[tick + 1]
        E_next = model.people.E[tick + 1]
        # Use S_next here since some S will have been removed by natural mortality and by human-to-human transmission
        local_s = np.round(local_frac * S_next).astype(S_next.dtype)
        new_infections = model.prng.binomial(local_s, -np.expm1(-Psi)).astype(S_next.dtype)
        S_next -= new_infections
        E_next += new_infections
        model.patches.incidence_env[tick + 1] += new_infections
        model.patches.incidence[tick + 1] += new_infections

        assert np.all(S_next >= 0), f"Negative susceptible populations at tick {tick + 1}.\n\t{S_next=}"

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield one Matplotlib figure of `Psi(t)` for the ten largest patches.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            The string label `"Environmental Transmission Rate"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Transmission Rate") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.Psi[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Environmental Transmission Rate")
        plt.legend()

        yield "Environmental Transmission Rate"
        return
