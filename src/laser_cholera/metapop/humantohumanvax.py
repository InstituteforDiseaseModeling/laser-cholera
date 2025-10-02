"""Human-to-human transmission rate."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class HumanToHumanVax:
    def __init__(self, model) -> None:
        self.model = model

        model.patches.add_vector_property("V1_incidence_hum", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("V2_incidence_hum", length=model.params.nticks + 1, dtype=np.int32, default=0)

        if not hasattr(model.patches, "incidence"):
            model.patches.add_vector_property("incidence", length=model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def check(self):
        assert hasattr(self.model, "patches"), "HumanToHuman: model needs to have a 'patches' attribute."
        assert hasattr(self.model.patches, "Lambda"), "HumanToHuman: model patches needs to have a 'Lambda' attribute."

        assert hasattr(self.model, "people"), "HumanToHuman: model needs to have a 'people' attribute."
        assert hasattr(self.model.people, "E"), "HumanToHuman: model people needs to have a 'E' (exposed) attribute."
        assert hasattr(self.model.people, "V1sus"), "HumanToHuman: model people needs to have a 'V1sus' (vaccinated 1 susceptible) attribute."
        assert hasattr(self.model.people, "V1inf"), "HumanToHuman: model people needs to have a 'V1inf' (vaccinated 1 infectious) attribute."
        assert hasattr(self.model.people, "V2sus"), "HumanToHuman: model people needs to have a 'V2sus' (vaccinated 2 susceptible) attribute."
        assert hasattr(self.model.people, "V2inf"), "HumanToHuman: model people needs to have a 'V2inf' (vaccinated 2 infectious) attribute."

        assert hasattr(self.model, "params"), "HumanToHuman: model needs to have a 'params' attribute."
        assert "tau_i" in self.model.params, "HumanToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter."

        return

    def __call__(self, model, tick: int) -> None:
        r"""
        Calculate the current human-to-human transmission rate per patch.

        $\Lambda_{j,t+1} = \frac {\beta^{hum}_{jt}((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{\forall i \neq j (\pi_{ij} \tau_j I_{it})}))^{\alpha_1}} {N^{\alpha_2}_{jt}}$

        """

        Lambda = model.patches.Lambda[tick + 1]
        E_next = model.people.E[tick + 1]

        # LambdaV1
        V1sus = model.people.V1sus[tick]
        local = np.round((1 - model.params.tau_i) * V1sus).astype(V1sus.dtype)
        new_infections = model.prng.binomial(local, -np.expm1(-Lambda)).astype(V1sus.dtype)
        V1sus_next = model.people.V1sus[tick + 1]
        V1inf_next = model.people.V1inf[tick + 1]
        V1sus_next -= new_infections
        V1inf_next += new_infections
        E_next += new_infections
        model.patches.V1_incidence_hum[tick + 1] += new_infections
        model.patches.incidence[tick + 1] += new_infections

        assert np.all(V1sus_next >= 0), f"V1sus' should not go negative ({tick=}\n\t{V1sus_next})"

        # LambdaV2
        V2sus = model.people.V2sus[tick]
        local = np.round((1 - model.params.tau_i) * V2sus).astype(V2sus.dtype)
        new_infections = model.prng.binomial(local, -np.expm1(-Lambda)).astype(V2sus.dtype)
        V2sus_next = model.people.V2sus[tick + 1]
        V2inf_next = model.people.V2inf[tick + 1]
        V2sus_next -= new_infections
        V2inf_next += new_infections
        E_next += new_infections
        model.patches.V2_incidence_hum[tick + 1] += new_infections
        model.patches.incidence[tick + 1] += new_infections

        assert np.all(V2sus_next >= 0), f"V2sus' should not go negative ({tick=}\n\t{V2sus_next})"

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        yield plot_helper(fig, "V1 Incidence (Human Transmission)", self.model.patches.V1_incidence_env, self.model.params.location_name)

        yield plot_helper(fig, "V2 Incidence (Human Transmission)", self.model.patches.V2_incidence_env, self.model.params.location_name)

        return


# TODO - copy-paste from envtohumanvax.py, should consider consolidating
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
        ax.plot(data[:, i], color)  # , label=f"{self.model.params.location_name[ipatch]}")
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

    return title
