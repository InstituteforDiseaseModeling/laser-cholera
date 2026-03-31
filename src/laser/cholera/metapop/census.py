import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Census:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "patches"), "Census: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("N", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(self.model, "params"), "Census: model needs to have a 'params' attribute."

        return

    def check(self):
        self(self.model, -1)  # a little hacky, but we need to set the initial population size

        return

    def __call__(self, model, tick: int) -> None:
        for compartment in ["S", "E", "Isym", "Iasym", "R", "V1", "V2"]:
            if hasattr(model.people, compartment):
                model.patches.N[tick + 1] += getattr(model.people, compartment)[tick + 1]

        assert np.all(model.patches.N[tick + 1] >= 0), "N' should not go negative"

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Census (Total Population)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.N[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Total Population")
        plt.legend()

        yield "Census (Total Population)"
        return
