"""Census compartment — derives total per-patch population from the SEIRV compartments.

Allocates `model.patches.N` (shape `(nticks + 1, npatches)`) and on
each tick sums the available compartments (`S`, `E`, `Isym`, `Iasym`,
`R`, `V1`, `V2`) into it. Other components (births, force-of-infection,
disease mortality) read `N` rather than recomputing the sum, so `Census`
must run *after* each compartment has written `[tick + 1]` but *before*
any downstream component that consumes `N`.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr


class Census:
    """Population-census component: maintains `patches.N` as the sum of all SEIRV compartments.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate the `N` per-patch population vector.

        Args:
            model: The `Model` instance. Must have `patches` and
                `params` set up.

        Raises:
            AttributeError: When `model.patches` or `model.params` is
                missing.
        """
        self.model = model

        check_attr(model, "patches", "Census: model needs to have a 'patches' attribute.")
        model.patches.add_vector_property("N", length=model.params.nticks + 1, dtype=np.int32, default=0)
        check_attr(self.model, "params", "Census: model needs to have a 'params' attribute.")

        return

    def check(self):
        """Seed `N[0]` by running `__call__` with `tick=-1`.

        Slightly hacky: `__call__` writes into `N[tick + 1]`, so
        `tick=-1` ends up writing the initial population from the
        compartments that have already been seeded by their own
        `__init__` methods. Avoids duplicating the sum logic.
        """
        self(self.model, -1)  # a little hacky, but we need to set the initial population size

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Sum the SEIRV compartments at `tick + 1` into `patches.N[tick + 1]`.

        Compartments that are not present on `model.people` (e.g.
        `V1` / `V2` if `Vaccinated` was excluded) are silently skipped,
        making the component robust to alternative pipelines.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.

        Raises:
            AssertionError: When the resulting per-patch total is
                negative (indicates an upstream compartment underflowed).
        """
        for compartment in ["S", "E", "Isym", "Iasym", "R", "V1", "V2"]:
            if hasattr(model.people, compartment):
                model.patches.N[tick + 1] += getattr(model.people, compartment)[tick + 1]

        assert np.all(model.patches.N[tick + 1] >= 0), "N' should not go negative"

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield one Matplotlib figure of `N(t)` for the ten largest patches.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            The string label `"Census (Total Population)"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Census (Total Population)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.N[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Total Population")
        plt.legend()

        yield "Census (Total Population)"
        return
