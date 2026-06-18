"""Environmental-reservoir component — tracks `W(t)` shedding from infectious people and decay.

Owns `model.patches.W` (the contaminated-water reservoir, one value per
patch per tick) and `model.patches.delta_jt` (per-patch decay rate
derived from `psi_jt` via a Beta CDF map). On each tick:

1. Decay: Poisson draw at rate `delta_jt[tick] * W`, clamped not to
    exceed the current reservoir.
2. Shedding from symptomatic cases: Poisson draw at `zeta_1 * Isym`,
    attenuated by `(1 - theta_j)` for WASH coverage.
3. Shedding from asymptomatic cases: Poisson draw at `zeta_2 * Iasym`,
    attenuated the same way.

`W` is consumed by
[`EnvToHuman`][laser.cholera.metapop.envtohuman.EnvToHuman] to drive
environmental transmission.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import beta

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key


class Environmental:
    """Environmental-reservoir component: maintains `patches.W` per tick.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate `W` / `delta_jt` and pre-compute the suitability-to-decay map.

        `delta_jt` is built once from `params.psi_jt` via
        [`map_suitability_to_decay`][laser.cholera.metapop.environmental.map_suitability_to_decay]
        (a Beta-CDF interpolator between `decay_days_short` and
        `decay_days_long`).

        Args:
            model: The `Model` instance. Must have `patches` and
                `params` with `psi_jt`, `decay_days_short`,
                `decay_days_long`, `decay_shape_1`, and `decay_shape_2`
                populated.

        Raises:
            AttributeError: When `model.patches` or `model.params` is
                missing.
            ValueError: When any of `params.psi_jt`, `decay_days_short`,
                `decay_days_long`, `decay_shape_1`, `decay_shape_2` is
                missing.
        """
        self.model = model

        check_attr(model, "patches", "Environmental: model needs to have a 'patches' attribute.")

        model.patches.add_vector_property("W", length=model.params.nticks + 1, dtype=np.float32, default=0.0)
        check_attr(model, "params", "Environmental: model needs to have a 'params' attribute.")
        check_key(model.params, "psi_jt", "Environmental: model params needs to have a 'psi_jt' (environmental contagion rate) parameter.")
        psi = model.params.psi_jt  # convenience
        # TODO - use newer laser_core with add_array_property and psi.shape
        model.patches.add_vector_property("delta_jt", length=psi.shape[0], dtype=np.float32, default=0.0)

        check_key(
            model.params,
            "decay_days_short",
            "Environmental: model params needs to have a 'decay_days_short' (maximum environmental decay) parameter.",
        )
        check_key(
            model.params, "decay_days_long", "Environmental: model params needs to have a 'decay_days_long' (minimum environmental decay) parameter."
        )
        check_key(
            self.model.params, "decay_shape_1", "Environmental: model params needs to have a 'decay_shape_1' (beta function parameter 1) parameter."
        )
        check_key(
            self.model.params, "decay_shape_2", "Environmental: model params needs to have a 'decay_shape_2' (beta function parameter 2) parameter."
        )

        model.patches.delta_jt[:, :] = map_suitability_to_decay(
            fast=model.params.decay_days_short,
            slow=model.params.decay_days_long,
            suitability=model.params.psi_jt,
            beta_a=model.params.decay_shape_1,
            beta_b=model.params.decay_shape_2,
        )

        return

    def check(self):
        """Validate `Isym` / `Iasym` are available and shedding parameters are present.

        Raises:
            AttributeError: When `model.people.Isym` or `.Iasym` is
                missing.
            ValueError: When `params.zeta_1`, `params.zeta_2`, or
                `params.theta_j` is missing.
        """
        check_attr(self.model, "people", "Environmental: model needs to have a 'people' attribute.")
        check_attr(self.model.people, "Isym", "Environmental: model people needs to have a 'Isym' (symptomatic) attribute.")
        check_attr(self.model.people, "Iasym", "Environmental: model people needs to have a 'Iasym' (asymptomatic) attribute.")
        check_key(self.model.params, "zeta_1", "Environmental: model params needs to have a 'zeta_1' (symptomatic shedding rate) parameter.")
        check_key(self.model.params, "zeta_2", "Environmental: model params needs to have a 'zeta_2' (asymptomatic shedding rate) parameter.")
        check_key(self.model.params, "theta_j", "Environmental: model params needs to have a 'theta_j' (fraction of population with WASH) attribute.")

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Advance `W` one tick: carry forward, decay, then shed from infectious cohorts.

        Decay is clamped not to exceed `W` so the reservoir never goes
        negative even on extreme Poisson draws. WASH coverage
        (`1 - theta_j`) attenuates the shed amount entering the reservoir.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.
        """
        W = model.patches.W[tick]
        W_next = model.patches.W[tick + 1]
        W_next[:] = W

        Isym = model.people.Isym[tick]
        Iasym = model.people.Iasym[tick]

        # -decay
        # Use np.minimum() to make sure we don't go negative
        decay = np.minimum(model.prng.poisson(model.patches.delta_jt[tick] * W), W).astype(W_next.dtype)
        W_next -= decay

        # +shedding from Isymptomatic
        shedding_sym = model.prng.poisson(model.params.zeta_1 * Isym).astype(W_next.dtype)
        W_next += (1 - model.params.theta_j) * shedding_sym

        # +shedding from Iasymptomatic
        shedding_asym = model.prng.poisson(model.params.zeta_2 * Iasym).astype(W_next.dtype)
        W_next += (1 - model.params.theta_j) * shedding_asym

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield three Matplotlib figures: reservoir trajectory, decay heatmap, suitability map.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            Three labels in order: `"Environmental Reservoir"`,
            `"Environmental Decay Rate"`, `"Suitability to Decay Mapping"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Reservoir") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.W[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Environmental Reservoir")
        plt.legend()

        yield "Environmental Reservoir"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Decay Rate") if fig is None else fig

        plt.imshow(self.model.patches.delta_jt.T, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Environmental Decay Rate")
        plt.xlabel("Tick")
        plt.ylabel("Patch")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Environmental Decay Rate"

        title = "Suitability to Decay Mapping"
        _fig = plt.figure(figsize=(12, 9), dpi=128, num=title) if fig is None else fig

        x = np.linspace(0, 1, self.model.params.psi_jt.shape[0])
        y = map_suitability_to_decay(
            self.model.params.decay_days_short,
            self.model.params.decay_days_long,
            x,
            self.model.params.decay_shape_1,
            self.model.params.decay_shape_2,
        )
        plt.plot(x, y, label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("psi_jt - suitability")
        plt.ylabel("delta_jt - decay rate")

        yield title
        return


# Put this is its own function so mapping plot is sure to use the same calculation.
def map_suitability_to_decay(fast: float, slow: float, suitability: np.ndarray, beta_a: float, beta_b: float) -> np.ndarray:
    r"""Map suitability to decay using a beta distribution.

    $$
    \delta_{jt} = \frac { 1 } { \text{days}_{short} + f( \psi_{jt}) ( \text{days}_{long}  - \text{days}_{short} ) }
    $$

    We use a parameterized beta distribution to map suitability values
    `[0, 1]` to `[0, 1]` in a, potentially, non-linear way.

    The resulting suitability factor determines a decay rate that is
    large when suitability is low — i.e. when suitability is 0, the
    decay rate is `1 / fast` and since `fast` is a short time or small
    number of days, `1 / fast` is relatively large — and small when
    suitability is high — i.e. when suitability is 1, the decay rate
    is `1 / slow` and since `slow` is a long time or larger number of
    days, `1 / slow` is relatively small.

    Args:
        fast: Fast decay time, in days.
        slow: Slow decay time, in days.
        suitability: Suitability values in `[0, 1]`.
        beta_a: Alpha parameter for the beta distribution.
        beta_b: Beta parameter for the beta distribution.

    Returns:
        Decay rates corresponding to the suitability values.
    """
    return 1.0 / (fast + beta.cdf(suitability, beta_a, beta_b) * (slow - fast))
