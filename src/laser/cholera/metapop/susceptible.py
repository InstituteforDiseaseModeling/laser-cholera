"""Susceptible compartment — owns the `S` per-patch state and the births/deaths flows that feed it.

Allocates `model.people.S` (shape `(nticks + 1, npatches)`), seeds it from
`params.S_j_initial`, and on each tick applies natural mortality (drawn
from `d_jt`) and Poisson-distributed births (drawn from `b_jt * N`).
Births enter `S` directly; deaths are recorded into
`model.patches.non_disease_deaths`.

Sits first in the default component pipeline so all subsequent
compartments can rely on `S[tick + 1]` being initialized to the carried-
forward `S` value when their own `__call__` runs.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model


class Component:
    """Marker base class for metapop compartment components.

    Currently empty: the metapop pipeline duck-types on
    `__init__(model)` / `check()` / `__call__(model, tick)` / `plot()`,
    so subclassing `Component` carries no behavior, only intent. Kept as
    an inheritance anchor for future shared default implementations.
    """


class Susceptible(Component):
    """Susceptible compartment: tracks `S_j(t)` and runs births / non-disease deaths each tick.

    Attributes:
        model: The parent `Model` instance; used to access `people`,
            `patches`, `params`, and the PRNG.
    """

    def __init__(self, model: "Model"):
        """Allocate the `S` state vector and seed it from `S_j_initial`.

        Adds `model.people.S` (shape `(nticks + 1, npatches)`) and
        `model.patches.births` (per-tick birth counts) as `int32` vector
        properties, then writes the initial susceptible counts from
        `params.S_j_initial` into `S[0]`.

        Args:
            model: The `Model` instance. Must already have `people`,
                `patches`, and `params` set up, and `params.S_j_initial`
                populated.

        Raises:
            AssertionError: When `model` is missing `people`, `patches`,
                or `params`, or `params` is missing `S_j_initial`.
        """
        self.model = model

        assert hasattr(model, "people"), "Susceptible: model needs to have an 'people' attribute."
        model.people.add_vector_property("S", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "patches"), "Susceptible: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("births", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Susceptible: model needs to have a 'params' attribute."
        assert "S_j_initial" in self.model.params, "Susceptible: model params needs to have a 'S_j_initial' parameter."
        model.people.S[0] = model.params.S_j_initial

        return

    def check(self):
        """Validate prerequisites that other components are expected to set up.

        Confirms `model.patches.N` (population census, set up by
        [`Census`][laser.cholera.metapop.census.Census]) and
        `params.b_jt` / `params.d_jt` (birth / non-disease-mortality
        matrices) are present. Lazily allocates
        `model.patches.non_disease_deaths` if no earlier component has
        already done so.

        Raises:
            AssertionError: When `model.patches.N`, `params.b_jt`, or
                `params.d_jt` is missing.
        """
        assert hasattr(self.model.patches, "N"), "Susceptible: model.patches needs to have a 'N' attribute."
        assert hasattr(self.model.params, "b_jt"), "Susceptible: model.params needs to have a 'b_jt' attribute."
        assert hasattr(self.model.params, "d_jt"), "Susceptible: model.params needs to have a 'd_jt' attribute."
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Advance `S` from `tick` to `tick + 1`: carry forward, kill, then birth.

        For each patch:

        1. `S[tick + 1] := S[tick]` (carry forward).
        2. Sample non-disease deaths from `Binomial(S, 1 - exp(-d_jt))`
            and subtract; bank into `patches.non_disease_deaths[tick]`.
        3. Sample births from `Poisson(N * b_jt[tick])` (where `N` is
            the current total patch population from `Census`) and add;
            record into `patches.births[tick]`.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick (0-indexed); `S[tick + 1]` is
                the slice this method writes to.

        Raises:
            AssertionError: When the post-deaths susceptible population
                goes negative (indicates a draw exceeded the population,
                which should be impossible for a `Binomial(S, p)` draw).
        """
        S_next = model.people.S[tick + 1]
        S_next[:] = model.people.S[tick]

        # natural mortality
        non_disease_deaths = model.prng.binomial(S_next, -np.expm1(-model.params.d_jt[tick])).astype(S_next.dtype)
        S_next -= non_disease_deaths
        model.patches.non_disease_deaths[tick] += non_disease_deaths

        assert np.all(S_next >= 0), f"Negative susceptible populations at tick {tick + 1}.\n\t{S_next=}"

        # births
        N = model.patches.N[tick]
        births = model.prng.poisson(N * model.params.b_jt[tick]).astype(S_next.dtype)
        S_next[:] += births
        model.patches.births[tick] = births

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield one Matplotlib figure showing `S(t)` for the ten largest patches.

        Patches are ranked by `params.S_j_initial` so the largest
        initial-susceptible populations are surfaced. Used by
        [`Model.visualize`][laser.cholera.metapop.model.Model] to
        generate the per-component PDF / on-screen plots.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into. If
                omitted, a fresh `(12, 9)` figure is created.

        Yields:
            The string label `"Susceptible"` (used by the visualizer as
            the figure title / PDF section heading).
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Susceptible") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.S[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Susceptible")
        plt.legend()

        yield "Susceptible"

        return
