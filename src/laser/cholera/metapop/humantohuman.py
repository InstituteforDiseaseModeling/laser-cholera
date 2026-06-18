"""Human-to-human transmission component.

Computes a per-tick, per-patch force-of-infection
[`Lambda`][laser.cholera.metapop.humantohuman.HumanToHuman.__call__]
from the symptomatic and asymptomatic infectious populations, mixed
through a gravity-model spatial-connectivity matrix (`pi_ij`, derived
from latitude/longitude) and modulated by a seasonality envelope
(`beta_jt_human`, derived from the `a_*_j` / `b_*_j` / `p` Fourier
coefficients). New `S -> E` transitions are sampled from
`Binomial(S_next, 1 - exp(-Lambda))` and recorded into
`patches.incidence_human` and the shared `patches.incidence`.
"""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from laser.cholera.metapop.utils import get_daily_seasonality
from laser.cholera.metapop.utils import get_pi_from_lat_long

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key

logger = logging.getLogger("laser.cholera")


class HumanToHuman:
    """Direct (human-to-human) transmission component for the metapop pipeline.

    Owns the per-patch force-of-infection vector `patches.Lambda`, the
    spatial-connectivity matrix `patches.pi_ij`, and the seasonality
    matrix `patches.beta_jt_human`.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate `Lambda`, build `pi_ij` from lat/long, and bake the seasonality matrix.

        `pi_ij` is computed once via
        [`get_pi_from_lat_long`][laser.cholera.metapop.utils.get_pi_from_lat_long]
        (gravity model), and `beta_jt_human` once via
        [`get_daily_seasonality`][laser.cholera.metapop.utils.get_daily_seasonality]
        (Fourier sum) — neither changes during the run.

        Args:
            model: The `Model` instance. Must have `patches`, `people`,
                and `params` set up, with `latitude`, `longitude`, the
                seasonality coefficients (`a_*_j`, `b_*_j`, `p`), and
                the mobility parameters (`mobility_omega`,
                `mobility_gamma`) populated.

        Raises:
            AttributeError: When `model.patches` is missing.
            ValueError: When any of the seasonality / mobility parameters
                (`latitude`, `longitude`, `mobility_omega`,
                `mobility_gamma`, `a_*_j`, `b_*_j`, `p`) is missing.
        """
        self.model = model

        check_attr(model, "patches", "HumanToHuman: model needs to have a 'patches' attribute.")
        model.patches.add_vector_property("Lambda", length=model.params.nticks + 1, dtype=np.float32, default=0.0)

        check_key(self.model.params, "latitude", "HumanToHuman: model params needs to have a 'latitude' (location latitude) parameter.")
        check_key(self.model.params, "longitude", "HumanToHuman: model params needs to have a 'longitude' (location longitude) parameter.")
        check_key(self.model.params, "mobility_omega", "HumanToHuman: model params needs to have a 'mobility_omega' (mobility) parameter.")
        check_key(self.model.params, "mobility_gamma", "HumanToHuman: model params needs to have a 'mobility_gamma' (mobility) parameter.")

        model.patches.add_array_property("pi_ij", (model.patches.count, model.patches.count), dtype=np.float32, default=0.0)
        model.patches.pi_ij[:, :] = get_pi_from_lat_long(model.params)

        check_key(self.model.params, "a_1_j", "HumanToHuman: model params needs to have a 'a_1_j' (seasonality) parameter.")
        check_key(self.model.params, "b_1_j", "HumanToHuman: model params needs to have a 'b_1_j' (seasonality) parameter.")
        check_key(self.model.params, "a_2_j", "HumanToHuman: model params needs to have a 'a_2_j' (seasonality) parameter.")
        check_key(self.model.params, "b_2_j", "HumanToHuman: model params needs to have a 'b_2_j' (seasonality) parameter.")
        check_key(self.model.params, "p", "HumanToHuman: model params needs to have a 'p' (seasonality pahse) parameter.")

        model.patches.add_array_property("beta_jt_human", (model.params.nticks, model.patches.count), dtype=np.float32, default=0.0)
        model.patches.beta_jt_human[:, :] = get_daily_seasonality(model.params)

        model.patches.add_vector_property("incidence_human", length=model.params.nticks + 1, dtype=np.int32, default=0)

        if not hasattr(model.patches, "incidence"):
            model.patches.add_vector_property("incidence", length=model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def check(self):
        """Validate the consumer-side compartments and transmission parameters.

        Asserts that `model.people` has `Isym`, `Iasym`, `S`, `E`;
        `model.patches.N` exists; and `params` provides `tau_i`,
        `beta_j0_hum`, `alpha_1`, and `alpha_2`.

        Raises:
            AttributeError: When `model`, `model.people` (with `Isym` /
                `Iasym` / `S` / `E`), or `model.patches.N` is missing.
            ValueError: When `params.tau_i`, `beta_j0_hum`, `alpha_1`, or
                `alpha_2` is missing.
        """
        check_attr(self.model, "people", "HumanToHuman: model needs to have a 'people' attribute.")
        check_attr(self.model.people, "Isym", "HumanToHuman: model people needs to have a 'Isym' (symptomatic) attribute.")
        check_attr(self.model.people, "Iasym", "HumanToHuman: model people needs to have a 'Iasym' (asymptomatic) attribute.")
        check_attr(self.model.people, "S", "HumanToHuman: model people needs to have a 'S' (susceptible) attribute.")
        check_attr(self.model.people, "E", "HumanToHuman: model people needs to have a 'E' (exposed) attribute.")

        check_attr(self.model.patches, "N", "HumanToHuman: model people needs to have a 'N' (current people) attribute.")

        check_attr(self.model, "params", "HumanToHuman: model needs to have a 'params' attribute.")
        check_key(self.model.params, "tau_i", "HumanToHuman: model params needs to have a 'tau_i' (emmigration probability) parameter.")
        check_key(
            self.model.params, "beta_j0_hum", "HumanToHuman: model params needs to have a 'beta_j0_hum' (baseline transmission rate) parameter."
        )

        check_key(self.model.params, "alpha_1", "HumanToHuman: model params needs to have an 'alpha_1' (numerator power) parameter.")
        check_key(self.model.params, "alpha_2", "HumanToHuman: model params needs to have an 'alpha_2' (denominator power) parameter.")

        return

    def __call__(self, model: "Model", tick: int) -> None:
        r"""Calculate the current human-to-human transmission rate per patch.

        $$
        \Lambda_{j,t+1} = \frac {\beta^{hum}_{jt}((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{\forall i \neq j (\pi_{ij} \tau_j I_{it})}))^{\alpha_1}} {N^{\alpha_2}_{jt}}
        $$
        """

        # LambdaS
        Lambda = model.patches.Lambda[tick + 1]
        Isym = model.people.Isym[tick]
        Iasym = model.people.Iasym[tick]
        N = model.patches.N[tick]
        S_next = model.people.S[tick + 1]
        E_next = model.people.E[tick + 1]

        total_i = Isym + Iasym
        local_frac = 1 - model.params.tau_i
        local_i = (local_frac * total_i).astype(Lambda.dtype)
        # This odd formulation (vector * matrix.T).T ensures that the result is indexed [src, dst] just like the matrix pi_ij
        immigrating_i = ((model.params.tau_i * total_i) * model.patches.pi_ij.T).T.sum(axis=0).astype(Lambda.dtype)
        effective_i = local_i + immigrating_i
        power_adjusted = np.power(effective_i, model.params.alpha_1).astype(Lambda.dtype)
        seasonality = (model.patches.beta_jt_human[tick, :]).astype(Lambda.dtype)
        adjusted = seasonality * power_adjusted
        denominator = np.power(N, model.params.alpha_2).astype(Lambda.dtype)
        rate = (adjusted / denominator).astype(Lambda.dtype)

        # TODO - check seasonality and power_adjusted for negative values so we don't have to do this
        if np.any(rate < 0.0):
            logger.debug(f"Negative transmission rate at tick {tick + 1}.\n\t{rate=}")
            rate = np.maximum(rate, 0.0)

        Lambda[:] = rate
        # Use S_next here since some S will have been removed by natural mortality and by environmental transmission
        local = np.round(local_frac * S_next).astype(S_next.dtype)
        if np.any(np.isnan(rate)):
            logger.debug(f"NaN transmission rate at tick {tick + 1}.\n\t{rate=}")
        new_infections = model.prng.binomial(local, -np.expm1(-rate)).astype(S_next.dtype)
        S_next -= new_infections
        E_next += new_infections
        model.patches.incidence_human[tick + 1] += new_infections
        model.patches.incidence[tick + 1] += new_infections

        assert np.all(S_next >= 0), f"Negative susceptible populations at tick {tick + 1}.\n\t{S_next=}"

        return

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield three Matplotlib figures: transmission rate, `pi_ij` heatmap, seasonality heatmap.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            Three labels in order:
            `"Human-to-Human Transmission Rate"`,
            `"Spatial Connectivity Matrix (pi_ij)"`,
            `"Seasonal Human-Human Transmission Factor by Location Over Time"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Human-to-Human Transmission Rate") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.Lambda[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Transmission Rate")
        plt.legend()

        yield "Human-to-Human Transmission Rate"

        # Spatial connectivity matrix
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Spatial Connectivity Matrix (pi_ij)") if fig is None else fig

        plt.imshow(self.model.patches.pi_ij, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(label="Connectivity")
        plt.xlabel("Destination Location Index")
        plt.xticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Source Location Index")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)
        plt.yticks(rotation=45, ha="right")

        yield "Spatial Connectivity Matrix (pi_ij)"

        # Seasonality factor by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Seasonal Human-Human Transmission Factor by Location Over Time") if fig is None else fig

        indices = np.argsort(self.model.params.latitude)[::-1]
        plt.imshow(self.model.patches.beta_jt_human[:, indices].T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Seasonal Factor")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=[self.model.params.location_name[i] for i in indices])

        yield "Seasonal Human-Human Transmission Factor by Location Over Time"

        return
