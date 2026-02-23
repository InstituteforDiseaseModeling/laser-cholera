from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Exposed:
    def __init__(self, model):
        self.model = model

        assert hasattr(model, "people"), "Exposed: model needs to have a 'people' attribute."
        model.people.add_vector_property("E", length=model.params.nticks + 1, dtype=np.int32, default=0)

        assert hasattr(self.model, "params"), "Exposed: model needs to have a 'params' attribute."
        assert "E_j_initial" in self.model.params, "Exposed: model params needs to have a 'E_j_initial' parameter."

        model.people.E[0] = model.params.E_j_initial

        return

    def check(self):
        # Don't bother checking for model.params, we did that in __init__()
        assert "iota" in self.model.params, "Exposed: model params needs to have a 'iota' (progression rate) parameter."
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        E_next = model.people.E[tick + 1]
        E = model.people.E[tick]
        E_next[:] = E

        # Do non-disease mortality first
        non_disease_deaths = model.prng.binomial(E, -np.expm1(-model.params.d_jt[tick])).astype(E_next.dtype)
        E_next -= non_disease_deaths
        model.patches.non_disease_deaths[tick] += non_disease_deaths

        return

    def plot(self, fig: Optional[Figure] = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Exposed") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.E[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Exposed")
        plt.legend()

        yield "Exposed"
        return
