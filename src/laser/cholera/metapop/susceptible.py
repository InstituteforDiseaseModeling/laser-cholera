import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Component: ...


class Susceptible(Component):
    def __init__(self, model):
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
        assert hasattr(self.model.patches, "N"), "Susceptible: model.patches needs to have a 'N' attribute."
        assert hasattr(self.model.params, "b_jt"), "Susceptible: model.params needs to have a 'b_jt' attribute."
        assert hasattr(self.model.params, "d_jt"), "Susceptible: model.params needs to have a 'd_jt' attribute."
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
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

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Susceptible") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.S[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Susceptible")
        plt.legend()

        yield "Susceptible"

        return
