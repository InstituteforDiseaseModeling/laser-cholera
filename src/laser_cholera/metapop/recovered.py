import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Recovered:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "people"), "Recovered: model needs to have a 'people' attribute."
        model.people.add_vector_property("R", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "params"), "Recovered: model needs to have a 'params' attribute."
        assert "R_j_initial" in model.params, "Recovered: model params needs to have a 'R_j_initial' (initial recovered population) parameter."

        model.people.R[0] = model.params.R_j_initial

        return

    def check(self):
        assert hasattr(self.model.people, "S"), "Recovered: model people needs to have a 'S' (susceptible) attribute."
        assert "d_jt" in self.model.params, "Recovered: model params needs to have a 'd_jt' (mortality rate) parameter."
        assert "epsilon" in self.model.params, "Recovered: model params needs to have a 'epsilon' (waning immunity rate) parameter."
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)
        return

    def __call__(self, model, tick: int) -> None:
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

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Recovered") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.R[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Recovered")
        plt.legend()

        yield "Recovered"
        return
