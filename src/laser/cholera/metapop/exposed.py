"""Exposed compartment — incubating individuals between infection and infectiousness.

Allocates `model.people.E` (shape `(nticks + 1, npatches)`), seeds it
from `params.E_j_initial`, and on each tick applies non-disease
mortality (drawn from `d_jt`). Progression from `E` to `Isym` / `Iasym`
is performed inside
[`Infectious`][laser.cholera.metapop.infectious.Infectious] using the
`iota` rate, so `Exposed.__call__` only handles the demographic decay.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key


class Exposed:
    """Exposed compartment: tracks `E_j(t)` and bleeds off non-disease deaths each tick.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model"):
        """Allocate the `E` state vector and seed it from `E_j_initial`.

        Args:
            model: The `Model` instance. Must already have `people` and
                `params` set up, with `params.E_j_initial` populated.

        Raises:
            AttributeError: When `model` is missing `people` or
                `params`.
            ValueError: When `params.E_j_initial` is missing.
        """
        self.model = model

        check_attr(model, "people", "Exposed: model needs to have a 'people' attribute.")
        model.people.add_vector_property("E", length=model.params.nticks + 1, dtype=np.int32, default=0)

        check_attr(self.model, "params", "Exposed: model needs to have a 'params' attribute.")
        check_key(self.model.params, "E_j_initial", "Exposed: model params needs to have a 'E_j_initial' parameter.")

        model.people.E[0] = model.params.E_j_initial

        return

    def check(self):
        """Validate the `iota` progression-rate parameter and ensure shared bookkeeping is allocated.

        Raises:
            ValueError: When `params.iota` is missing.
        """
        # Don't bother checking for model.params, we did that in __init__()
        check_key(self.model.params, "iota", "Exposed: model params needs to have a 'iota' (progression rate) parameter.")
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)
        # PERF: install `model.patches.non_disease_death_prob_jt` (the
        # `1 - exp(-d_jt)` cache) idempotently if no earlier component has.
        # Lets each consumer be exercised in isolation; the cache is built
        # exactly once across the pipeline.
        if not hasattr(self.model.patches, "non_disease_death_prob_jt"):
            self.model.patches.add_vector_property("non_disease_death_prob_jt", length=self.model.params.d_jt.shape[0], dtype=np.float32, default=0.0)
            self.model.patches.non_disease_death_prob_jt[:] = -np.expm1(-self.model.params.d_jt)

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Advance `E` from `tick` to `tick + 1`: carry forward, then remove non-disease deaths.

        Progression `E -> Isym / Iasym` (governed by `iota` and `sigma`)
        is handled by [`Infectious`][laser.cholera.metapop.infectious.Infectious];
        this method only applies natural mortality from `d_jt`.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.
        """
        E_next = model.people.E[tick + 1]
        E = model.people.E[tick]
        E_next[:] = E

        # Do non-disease mortality first
        # PERF: `model.patches.non_disease_death_prob_jt[tick]` is the pre-computed
        # `-np.expm1(-model.params.d_jt[tick])` cached by Susceptible.check().
        # non_disease_deaths = model.prng.binomial(E, -np.expm1(-model.params.d_jt[tick])).astype(E_next.dtype)
        non_disease_deaths = model.prng.binomial(E, model.patches.non_disease_death_prob_jt[tick]).astype(E_next.dtype)
        E_next -= non_disease_deaths
        model.patches.non_disease_deaths[tick] += non_disease_deaths

        return

    def plot(self, fig: Optional[Figure] = None) -> Iterator[str]:  # pragma: no cover
        """Yield a single Matplotlib figure of `E(t)` for the ten largest patches.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            The string label `"Exposed"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Exposed") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.E[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Exposed")
        plt.legend()

        yield "Exposed"
        return
