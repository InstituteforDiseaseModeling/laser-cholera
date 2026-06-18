"""Vaccinated compartment — single-dose and two-dose protected populations.

Owns:

- `model.people.V1` — one-dose vaccinated population.
- `model.people.V2` — two-dose vaccinated population.
- `model.patches.dose_one_doses` / `dose_two_doses` — per-tick dose
  counts (recorded on the day delivered for direct alignment with
  `nu_1_jt` / `nu_2_jt`).

Per-tick flow:

1. Carry `V1` / `V2` forward, then subtract non-disease deaths (`d_jt`).
2. Wane `V1 -> S` at rate `omega_1` and `V2 -> S` at rate `omega_2`.
3. Deliver second doses (`nu_2_jt[tick]`) by moving the effective
   fraction (`phi_2`) from `V1 -> V2`. Done before first doses so an
   individual cannot move `S -> V1 -> V2` within a single tick.
4. Deliver first doses (`nu_1_jt[tick]`) drawn proportionally from the
   eligible source compartments (default: `S`, `E`, `Isym`, `Iasym`,
   `R`; overridable via `params.nu_jt_sources`); the effective fraction
   (`phi_1`) lands in `V1`, with the remainder staying in the source
   compartment (i.e. the dose failed to take).
"""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key

logger = logging.getLogger("laser.cholera")


class Vaccinated:
    """Vaccinated compartment: tracks `V1(t)` / `V2(t)` and the per-tick dose flows.

    Attributes:
        model: The parent `Model` instance.
        sources: Names of `model.people` compartments eligible to draw
            first doses from (default `["S", "E", "Isym", "Iasym", "R"]`,
            overridable via `params.nu_jt_sources`).
    """

    def __init__(self, model: "Model") -> None:
        """Allocate `V1` / `V2` and the dose-bookkeeping vectors; seed from `V*_j_initial`.

        Args:
            model: The `Model` instance. Must have `people`, `patches`,
                and `params`, with `params.V1_j_initial` and
                `params.V2_j_initial` populated.

        Raises:
            AttributeError: When `model` is missing `people` or
                `patches`.
            ValueError: When `params.V1_j_initial` or
                `params.V2_j_initial` is missing.
        """
        self.model = model

        check_attr(model, "people", "Vaccinated: model needs to have a 'people' attribute.")
        model.people.add_vector_property("V1", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V2", length=model.params.nticks + 1, dtype=np.int32, default=0)
        # We will track doses on the date (tick) given to more easily match nu_1_jt and nu_2_jt.
        model.patches.add_vector_property("dose_one_doses", length=model.params.nticks, dtype=np.int32, default=0)
        model.patches.add_vector_property("dose_two_doses", length=model.params.nticks, dtype=np.int32, default=0)
        check_key(
            model.params,
            "V1_j_initial",
            "Vaccinated: model params needs to have a 'V1_j_initial' (initial one dose vaccinated population) parameter.",
        )
        check_key(
            model.params,
            "V2_j_initial",
            "Vaccinated: model params needs to have a 'V2_j_initial' (initial two dose vaccinated population) parameter.",
        )
        model.people.V1[0] = model.params.V1_j_initial
        model.people.V2[0] = model.params.V2_j_initial

        self.sources = model.params.nu_jt_sources if "nu_jt_sources" in model.params else ["S", "E", "Isym", "Iasym", "R"]

        return

    def check(self):
        """Validate dose-related parameters and the source compartments.

        Asserts that `model.people.S` and `model.people.E` exist and
        that `params` provides `phi_1`, `phi_2`, `omega_1`, `omega_2`,
        `nu_1_jt`, `nu_2_jt`, and `d_jt`.

        Raises:
            AttributeError: When `model.people.S` or `.E` is missing.
            ValueError: When any of `params.phi_1`, `phi_2`, `omega_1`,
                `omega_2`, `nu_1_jt`, `nu_2_jt`, `d_jt` is missing.
        """
        check_attr(self.model.people, "S", "Vaccinated: model people needs to have a 'S' (susceptible) attribute.")
        check_attr(self.model.people, "E", "Vaccinated: model people needs to have a 'E' (exposed) attribute.")

        check_key(self.model.params, "phi_1", "Vaccinated: model params needs to have a 'phi_1' parameter.")
        check_key(self.model.params, "phi_2", "Vaccinated: model params needs to have a 'phi_2' parameter.")
        check_key(self.model.params, "omega_1", "Vaccinated: model params needs to have a 'omega_1' parameter.")
        check_key(self.model.params, "omega_2", "Vaccinated: model params needs to have a 'omega_2' parameter.")
        check_key(self.model.params, "nu_1_jt", "Vaccinated: model params needs to have a 'nu_1_jt' parameter.")
        check_key(self.model.params, "nu_2_jt", "Vaccinated: model params needs to have a 'nu_2_jt' parameter.")

        check_key(self.model.params, "d_jt", "Vaccinated: model.params needs to have a 'd_jt' attribute.")

        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)
        # PERF: install `model.patches.non_disease_death_prob_jt` (the
        # `1 - exp(-d_jt)` cache) idempotently if no earlier component has.
        # Lets each consumer be exercised in isolation; the cache is built
        # exactly once across the pipeline.
        if not hasattr(self.model.patches, "non_disease_death_prob_jt"):
            self.model.patches.add_vector_property("non_disease_death_prob_jt", length=self.model.params.d_jt.shape[0], dtype=np.float32, default=0.0)
            self.model.patches.non_disease_death_prob_jt[:] = -np.expm1(-self.model.params.d_jt)

        # PERF: cache `1 - exp(-omega_{1,2})` so the per-tick waning
        # probabilities are not recomputed every call.
        self._omega_1_prob = -np.expm1(-self.model.params.omega_1)
        self._omega_2_prob = -np.expm1(-self.model.params.omega_2)

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Advance `V1` / `V2` one tick: deaths, waning, then dose deliveries.

        Doses scheduled for this tick are clamped against the available
        donor population (warning logged at `DEBUG` if the schedule
        exceeds capacity). Only the `phi_*`-effective fraction transits
        to the vaccinated compartment; ineffective doses leave the
        recipient in their original compartment.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.
        """
        V1 = model.people.V1[tick]
        V1_next = model.people.V1[tick + 1]
        V2 = model.people.V2[tick]
        V2_next = model.people.V2[tick + 1]
        S_next = model.people.S[tick + 1]

        # propagate the current values forward (V(t+1) = V(t) + ∆V)
        V1_next[:] = V1
        V2_next[:] = V2

        # -natural mortality
        # PERF: pre-computed `-np.expm1(-d_jt)` cached as model.patches.non_disease_death_prob_jt.
        # non_disease_deaths = model.prng.binomial(V1_next, -np.expm1(-model.params.d_jt[tick])).astype(V1_next.dtype)
        non_disease_deaths = model.prng.binomial(V1_next, model.patches.non_disease_death_prob_jt[tick]).astype(V1_next.dtype)
        V1_next -= non_disease_deaths
        ndd_next = model.patches.non_disease_deaths[tick]
        ndd_next += non_disease_deaths

        # non_disease_deaths = model.prng.binomial(V2_next, -np.expm1(-model.params.d_jt[tick])).astype(V2_next.dtype)
        non_disease_deaths = model.prng.binomial(V2_next, model.patches.non_disease_death_prob_jt[tick]).astype(V2_next.dtype)
        V2_next -= non_disease_deaths
        ndd_next += non_disease_deaths

        # -waning immunity
        # PERF: pre-computed `-np.expm1(-omega_{1,2})` cached as self._omega_{1,2}_prob.
        # waned = model.prng.binomial(V1_next, -np.expm1(-model.params.omega_1)).astype(V1_next.dtype)
        waned = model.prng.binomial(V1_next, self._omega_1_prob).astype(V1_next.dtype)
        V1_next -= waned
        S_next += waned  # waned return to Susceptible

        # waned = model.prng.binomial(V2_next, -np.expm1(-model.params.omega_2)).astype(V2_next.dtype)
        waned = model.prng.binomial(V2_next, self._omega_2_prob).astype(V2_next.dtype)
        V2_next -= waned
        S_next += waned  # waned return to Susceptible

        # We will do _second_ dose distribution first so we don't vaccine people
        # "twice on the same day", i.e. move from S -> V1 -> V2 in a single tick
        # -second dose recipients
        if any(model.params.nu_2_jt[tick]):
            new_second_doses_delivered = np.round(model.params.nu_2_jt[tick]).astype(V2_next.dtype)
            if np.any(new_second_doses_delivered > V1_next):
                logger.debug(f"WARNING: new_second_doses_delivered > V1 ({tick=}\n\t{new_second_doses_delivered=}\n\t{V1=})")
                new_second_doses_delivered = np.minimum(new_second_doses_delivered, V1_next)
            model.patches.dose_two_doses[tick] = new_second_doses_delivered

            # effective doses
            newly_immunized = np.round(model.params.phi_2 * new_second_doses_delivered).astype(V2_next.dtype)
            # just move the effective doses, leave ineffective doses in V1
            V1_next -= newly_immunized
            V2_next += newly_immunized

        # +newly vaccinated (successful take)
        if any(model.params.nu_1_jt[tick]):
            new_first_doses_delivered = np.round(model.params.nu_1_jt[tick]).astype(V1_next.dtype)

            # Create a "matrix" of columns from model.people S, E, Isym, Iasym, and R with a row for each node
            compartments_next = [getattr(model.people, compartment)[tick + 1] for compartment in self.sources if hasattr(model.people, compartment)]
            pop_matrix = np.column_stack(compartments_next)

            # Sum each row by column to determine the available population for vaccination
            available_pop = pop_matrix.sum(axis=1)

            # Limit actual doses delivered to the available population by node
            if np.any(new_first_doses_delivered > available_pop):
                logger.debug(f"WARNING: new_first_doses_delivered > available_pop ({tick=})")
                for index in np.nonzero(new_first_doses_delivered > available_pop)[0]:
                    logger.debug(
                        f"\t{model.params.location_name[index]}: doses {new_first_doses_delivered[index]} > {available_pop[index]} available"
                    )
                new_first_doses_delivered = np.minimum(new_first_doses_delivered, available_pop)
            model.patches.dose_one_doses[tick] = new_first_doses_delivered

            # make sure available_pop is at least 1 to prevent divide by zero below
            np.maximum(available_pop, 1, out=available_pop)

            # Determine doses delivered to each sub-population by its fractional part of the total node population
            # Attenuate actual doses delivered by model.params.phi_1
            # Decrement each source sub-population by the effectively delivered doses
            # Increment V1_next by the total number of effectively delivered doses for each node
            total_newly_immunized = np.zeros_like(V1_next)
            for col_idx, compartment_next in enumerate(compartments_next):
                fraction = pop_matrix[:, col_idx] / available_pop
                compartment_doses = np.round(new_first_doses_delivered * fraction).astype(V1_next.dtype)
                effective_doses = np.round(model.params.phi_1 * compartment_doses).astype(V1_next.dtype)
                compartment_next -= effective_doses
                total_newly_immunized += effective_doses
            V1_next += total_newly_immunized

        return

    def plot(self, fig: Figure | None = None) -> Iterator[str]:  # pragma: no cover
        """Yield two Matplotlib figures: `V1(t)` and `V2(t)` for the ten largest patches.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            Two labels in order: `"Vaccinated (One Dose)"`,
            `"Vaccinated (Two Doses)"`.
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Vaccinated (One Dose)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.V1[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Vaccinated (One Dose)")
        plt.legend()

        yield "Vaccinated (One Dose)"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Vaccinated (Two Doses)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.V2[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Vaccinated (Two Doses)")
        plt.legend()

        yield "Vaccinated (Two Doses)"
        return
