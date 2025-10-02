"""Environmental transmission rate."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class EnvToHumanVax:
    def __init__(self, model) -> None:
        self.model = model

        model.patches.add_vector_property("V1_incidence_env", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("V2_incidence_env", length=model.params.nticks + 1, dtype=np.int32, default=0)

        if not hasattr(model.people, "incidence"):
            model.patches.add_vector_property("incidence", length=model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def check(self):
        assert hasattr(self.model, "people"), "EnvToHuman: model needs to have a 'people' attribute."
        assert hasattr(self.model.people, "E"), "EnvToHuman: model people needs to have a 'E' (exposed) attribute."
        assert hasattr(self.model.people, "V1sus"), "EnvToHuman: model people needs to have a 'V1sus' (vaccinated 1 susceptible) attribute."
        assert hasattr(self.model.people, "V1inf"), "EnvToHuman: model people needs to have a 'V1inf' (vaccinated 1 infected) attribute."
        assert hasattr(self.model.people, "V2sus"), "EnvToHuman: model people needs to have a 'V2sus' (vaccinated 2 susceptible) attribute."
        assert hasattr(self.model.people, "V2inf"), "EnvToHuman: model people needs to have a 'V2inf' (vaccinated 2 infected) attribute."

        assert hasattr(self.model, "patches"), "EnvToHuman: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "Psi"), "EnvToHuman: model patches needs to have a 'Psi' (environmental transmission rate) attribute."

        assert hasattr(self.model, "params"), "EnvToHuman: model needs to have a 'params' attribute."
        assert "tau_i" in self.model.params, "EnvToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        Psi = model.patches.Psi[tick + 1]
        tau_i = model.params.tau_i
        local_frac = 1 - tau_i

        E_next = model.people.E[tick + 1]

        # PsiV1
        V1sus = model.people.V1sus[tick]
        V1sus_next = model.people.V1sus[tick + 1]
        V1inf_next = model.people.V1inf[tick + 1]

        local_v1 = np.round(local_frac * V1sus).astype(V1sus.dtype)
        new_infections_v1 = model.prng.binomial(local_v1, -np.expm1(-Psi)).astype(V1sus.dtype)

        V1sus_next -= new_infections_v1
        assert np.all(V1sus_next >= 0), f"V1sus' should not go negative ({tick=}\n\t{V1sus_next})"

        V1inf_next += new_infections_v1
        E_next += new_infections_v1
        model.patches.V1_incidence_env[tick + 1] += new_infections_v1
        model.patches.incidence[tick + 1] += new_infections_v1

        # PsiV2
        V2sus = model.people.V2sus[tick]
        V2sus_next = model.people.V2sus[tick + 1]
        V2inf_next = model.people.V2inf[tick + 1]

        local_v2 = np.round(local_frac * V2sus).astype(V2sus.dtype)
        new_infections_v2 = model.prng.binomial(local_v2, -np.expm1(-Psi)).astype(V2sus.dtype)

        V2sus_next -= new_infections_v2
        assert np.all(V2sus_next >= 0), f"V2sus' should not go negative ({tick=}\n\t{V2sus_next})"

        V2inf_next += new_infections_v2
        E_next += new_infections_v2
        model.patches.V2_incidence_env[tick + 1] += new_infections_v2
        model.patches.incidence[tick + 1] += new_infections_v2

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        plot_helper(fig, "V1 Incidence (Environmental Transmission)", self.model.patches.V1_incidence_env, self.model.params.location_name)

        yield "V1 Incidence (Environmental Transmission)"

        plot_helper(fig, "V2 Incidence (Environmental Transmission)", self.model.patches.V2_incidence_env, self.model.params.location_name)

        yield "V2 Incidence (Environmental Transmission)"
        return


def plot_helper(fig, title, data, names):  # pragma: no cover
    _fig = plt.figure(num=title, figsize=(12, 9), dpi=128) if fig is None else fig
    rows, cols = 10, 4
    x_axes_refs = [None] * cols
    y_axes_refs = [None] * rows
    ymin, ymax = 0, max(data.max(), 100)  # avoid warning about ylim below
    for i in range(data.shape[1]):
        row = i // cols
        col = i % cols

        sharex = x_axes_refs[col]
        sharey = y_axes_refs[row]
        ax = _fig.add_subplot(rows, cols, i + 1, sharex=sharex, sharey=sharey)
        color = ["green", "red"][np.any(data[:, i] > 0)]
        ax.plot(data[:, i], color)  # , label=f"self.model.params.location_name[i]")
        ax.set_ylim(ymin, ymax)
        ax.set_title(f"{names[i]}", fontsize=8)

        show_xticks = row == rows - 1
        show_yticks = col == 0
        ax.tick_params(axis="x", which="both", bottom=show_xticks, labelbottom=show_xticks)
        ax.tick_params(axis="y", which="both", left=show_yticks, labelleft=show_yticks)

        x_axes_refs[col] = ax if x_axes_refs[col] is None else x_axes_refs[col]
        y_axes_refs[row] = ax if y_axes_refs[row] is None else y_axes_refs[row]

    _fig.text(0.5, 0.04, "Ticks", ha="center", fontsize=9)  # bottom center
    _fig.text(0.04, 0.5, "Incidence", va="center", rotation="vertical", fontsize=9)  # left center
    plt.tight_layout(rect=[0.06, 0.06, 1, 1])  # leave space for global labels

    return
