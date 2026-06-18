"""Test-only pipeline component that wipes out the infectious cohorts on tick 1.

This module lives under `tests/` rather than `src/laser/cholera/` because
`Eradication` is exclusively a diagnostic / fixture component — it is
not imported by any production code path. It is consumed by
[`tests/test_environmental.py`] and [`tests/test_envtohuman.py`] to
construct "what does the pipeline do when no infection survives
beyond tick 1?" baselines for the environmental-transmission
components.

pytest's default discovery pattern is `test_*.py` / `*_test.py` /
`tests.py`, so this file is not collected as a test module; the test
files that need it `import` it directly. Both the project's `tox.ini`
(`PYTHONPATH={toxinidir}/tests`) and pytest's default `prepend`
import mode put `tests/` on `sys.path`, so a bare
`from eradication import Eradication` resolves in both contexts.
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
        """No-op: `Eradication` contributes no figures to the visualization output.

        Implemented as `yield from ()` so this remains a generator
        function (matching the component-plot protocol expected by
        [`Model.visualize`][laser.cholera.metapop.model.Model.visualize])
        but produces zero items. A bare `yield` here would emit `None`
        and produce a phantom blank PDF page.

        Args:
            fig: Optional existing Matplotlib `Figure` (unused).

        Yields:
            Nothing; the generator is empty.
        """
        yield from ()
