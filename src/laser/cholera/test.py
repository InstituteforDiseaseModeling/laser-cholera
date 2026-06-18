"""Diagnostic / test components — degenerate scenarios for stress-checking the pipeline.

Currently holds [`Eradication`][laser.cholera.test.Eradication], a
component that wipes the infectious populations on tick 1 to verify
the rest of the pipeline gracefully handles a "no infection ever
spreads" run.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model


class Eradication:
    """Diagnostic component: at tick 1, move all infectious individuals back to `S`.

    Useful for confirming the pipeline behaves correctly when the
    disease is eliminated mid-run (no NaNs, no negative populations,
    likelihood code handles the all-zero observation series, etc.).

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Register the component on `model` (no state is allocated).

        Args:
            model: The `Model` instance.
        """
        self.model = model

        return

    def check(self):
        """No-op: `Eradication` has no prerequisites."""
        return

    def __call__(self, model: "Model", tick: int) -> None:
        """On tick 1, fold all symptomatic and asymptomatic infections back into `S`.

        Idempotent on subsequent ticks. Intended for diagnostic runs
        only — do not use in production scenarios.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.
        """
        if tick == 1:
            model.people.S[tick] += model.people.Isym[tick] + model.people.Iasym[tick]
            model.people.Isym[tick] = 0
            model.people.Iasym[tick] = 0

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """No-op generator to satisfy the component-plot protocol.

        Args:
            fig: Optional existing Matplotlib `Figure` (unused).

        Yields:
            A single `None`.
        """
        yield
        return
