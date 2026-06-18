"""Recovered compartment — natural-immunity population that wanes back to susceptible.

Allocates `model.people.R` (shape `(nticks + 1, npatches)`), seeds it
from `params.R_j_initial`, and on each tick applies non-disease
mortality (`d_jt`) and waning immunity (`epsilon`). Waned individuals
are funneled directly into `S[tick + 1]`. Recovery inflows into `R` are
managed by [`Infectious`][laser.cholera.metapop.infectious.Infectious]
via the `gamma_1` / `gamma_2` rates.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model


class Recovered:
    """Recovered compartment: tracks `R_j(t)` and lets natural immunity wane back to `S`.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate the `R` state vector and seed it from `R_j_initial`.

        Args:
            model: The `Model` instance. Must have `people` and `params`
                set up, with `params.R_j_initial` populated.

        Raises:
            AssertionError: When `model` is missing `people` / `params`
                or `params` is missing `R_j_initial`.
        """
        self.model = model

        assert hasattr(model, "people"), "Recovered: model needs to have a 'people' attribute."
        model.people.add_vector_property("R", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "params"), "Recovered: model needs to have a 'params' attribute."
        assert "R_j_initial" in model.params, "Recovered: model params needs to have a 'R_j_initial' (initial recovered population) parameter."

        model.people.R[0] = model.params.R_j_initial

        return

    def check(self):
        """Validate that `S` (waning destination), `d_jt`, and `epsilon` are available.

        Raises:
            AssertionError: When `model.people.S`, `params.d_jt`, or
                `params.epsilon` is missing.
        """
        assert hasattr(self.model.people, "S"), "Recovered: model people needs to have a 'S' (susceptible) attribute."
        assert "d_jt" in self.model.params, "Recovered: model params needs to have a 'd_jt' (mortality rate) parameter."
        assert "epsilon" in self.model.params, "Recovered: model params needs to have a 'epsilon' (waning immunity rate) parameter."
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)
        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Advance `R` from `tick` to `tick + 1`: carry forward, subtract deaths, wane to `S`.

        Waning is computed against the post-deaths cohort so dying-then-
        waning individuals are not counted twice.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.

        Raises:
            AssertionError: When the post-deaths-and-waning recovered
                population goes negative.
        """
        R = model.people.R[tick]
        R_next = model.people.R[tick + 1]
        S_next = model.people.S[tick + 1]

        R_next += R

        # natural mortality
        non_disease_deaths = model.prng.binomial(R, -np.expm1(-model.params.d_jt[tick])).astype(R_next.dtype)
        R_next -= non_disease_deaths
        model.patches.non_disease_deaths[tick] += non_disease_deaths

        # waning natural immunity - don't include those removed by natural mortality
        waned = model.prng.binomial(R - non_disease_deaths, -np.expm1(-model.params.epsilon)).astype(R_next.dtype)
        R_next -= waned
        S_next += waned

        assert np.all(R_next >= 0), f"Negative recovered populations at tick {tick + 1}.\n\t{R_next=}"

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield one Matplotlib figure of `R(t)` for the ten largest patches.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            The string label `"Recovered"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Recovered") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.R[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Recovered")
        plt.legend()

        yield "Recovered"
        return
