import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser.cholera.calc_model_likelihood import calc_model_likelihood


class Analyzer:
    def __init__(self, model) -> None:
        self.model = model

        return

    def check(self):
        return

    def __call__(self, model, tick: int) -> None:
        """Calculate log likelihood on the final tick."""
        # If model.params.calc_likelihood is True, calculate the log likelihood on the final tick.
        if tick == model.params.nticks - 1:
            if ("calc_likelihood" in model.params) and model.params.calc_likelihood:
                # Use the smaller of reported cases or the number of timesteps (not including the initial state)
                nreports = min(model.params.reported_cases.shape[1], model.patches.incidence.shape[0] - 1)
                try:
                    optional = {
                        key: model.params.key
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
                        ]
                        if key in model.params
                    }
                    model.log_likelihood = calc_model_likelihood(
                        obs_cases=model.params.reported_cases[:, :nreports],
                        est_cases=model.results.reported_cases[:, :nreports],
                        obs_deaths=model.params.reported_deaths[:, :nreports],
                        est_deaths=model.patches.reported_deaths[:, :nreports],
                        **optional,
                    )
                except ValueError as e:
                    print(f"Error calculating log likelihood: {e}")
                    model.log_likelihood = -np.inf
            else:
                model.log_likelihood = np.nan

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
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
