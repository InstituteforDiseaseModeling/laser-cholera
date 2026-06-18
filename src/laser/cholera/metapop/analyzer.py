"""End-of-run model-likelihood analyzer.

On the final tick, calls
[`calc_model_likelihood`][laser.cholera.calc_model_likelihood.calc_model_likelihood]
with observed and estimated reported cases / deaths, plus any optional
weight / sigma / epidemic-peak parameters available on `model.params`,
and stashes the scalar result on `model.log_likelihood`. Gated on
`params.calc_likelihood`; if disabled, `model.log_likelihood` is set
to `nan`; if the likelihood call raises `ValueError`, it's set to
`-inf` so downstream code (calibration sweeps, etc.) can detect the
failure without a try/except at every call site.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser.cholera.calc_model_likelihood import calc_model_likelihood

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model


class Analyzer:
    """Final-tick likelihood evaluator for calibration / scoring workflows.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Register the analyzer on the model (no state is allocated).

        Args:
            model: The `Model` instance.
        """
        self.model = model

        return

    def check(self):
        """No-op: `Analyzer` has no prerequisites that need pre-tick validation.

        The likelihood call validates its own inputs on the final tick.
        """
        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Calculate log likelihood on the final tick."""
        # If model.params.calc_likelihood is True, calculate the log likelihood on the final tick.
        if tick == model.params.nticks - 1:
            if ("calc_likelihood" in model.params) and model.params.calc_likelihood:
                # Use the smaller of reported cases or the number of timesteps (not including the initial state)
                nreports = min(model.params.reported_cases.shape[1], model.patches.incidence.shape[0] - 1)
                try:
                    optional = {
                        key: model.params[key]
                        for key in [
                            "weight_cases",
                            "weight_deaths",
                            "weights_time",
                            "weights_location",
                            "nb_k_min_cases",
                            "nb_k_min_deaths",
                            "weight_peak_timing",
                            "weight_peak_magnitude",
                            "weight_cumulative_total",
                            "weight_wis",
                            "sigma_peak_time",
                            "sigma_peak_log",
                            "epidemic_peaks",
                            "date_start",
                            "date_stop",
                        ]
                        if key in model.params
                    }

                    model.log_likelihood = calc_model_likelihood(
                        obs_cases=model.params.reported_cases[:, :nreports],
                        est_cases=model.results.reported_cases[:, :nreports],
                        obs_deaths=model.params.reported_deaths[:, :nreports],
                        est_deaths=model.results.reported_deaths[:, :nreports],
                        **optional,
                    )
                except ValueError as e:
                    print(f"Error calculating log likelihood: {e}")
                    model.log_likelihood = -np.inf
            else:
                model.log_likelihood = np.nan

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield one Matplotlib figure overlaying every SEIRV channel for the largest patch.

        Pulls the R-transposed series from `model.results` (so channels
        are indexed `[patch, tick]`) and plots all of `S`, `Isym`,
        `Iasym`, `R`, `V1`, `V2` for the single largest-by-initial-S
        patch.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            The string label `"SIRV Trajectories (Largest Patch)"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="SIRV Trajectories (Largest Patch)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-1:]:
            for channel in ["S", "Isym", "Iasym", "R", "V1", "V2"]:
                # Transpose ticks (:) and location since results are transposed for R users.
                plt.plot(getattr(self.model.results, channel)[ipatch, :], label=f"{channel}")

        plt.xlabel("Tick")
        plt.ylabel("Population")
        plt.legend()

        yield "SIRV Trajectories (Largest Patch)"
        return
