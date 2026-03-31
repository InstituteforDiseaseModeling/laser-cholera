import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class Vaccinated:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "people"), "Vaccinated: model needs to have a 'people' attribute."
        model.people.add_vector_property("V1", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V2", length=model.params.nticks + 1, dtype=np.int32, default=0)
        # We will track doses on the date (tick) given to more easily match nu_1_jt and nu_2_jt.
        model.patches.add_vector_property("dose_one_doses", length=model.params.nticks, dtype=np.int32, default=0)
        model.patches.add_vector_property("dose_two_doses", length=model.params.nticks, dtype=np.int32, default=0)
        assert "V1_j_initial" in model.params, (
            "Vaccinated: model params needs to have a 'V1_j_initial' (initial one dose vaccinated population) parameter."
        )
        assert "V2_j_initial" in model.params, (
            "Vaccinated: model params needs to have a 'V2_j_initial' (initial two dose vaccinated population) parameter."
        )
        model.people.V1[0] = model.params.V1_j_initial
        model.people.V2[0] = model.params.V2_j_initial

        self.sources = model.params.nu_jt_sources if "nu_jt_sources" in model.params else ["S", "E", "Isym", "Iasym", "R"]

        return

    def check(self):
        assert hasattr(self.model.people, "S"), "Vaccinated: model people needs to have a 'S' (susceptible) attribute."
        assert hasattr(self.model.people, "E"), "Vaccinated: model people needs to have a 'e' (exposed) attribute."

        assert "phi_1" in self.model.params, "Vaccinated: model params needs to have a 'phi_1' parameter."
        assert "phi_2" in self.model.params, "Vaccinated: model params needs to have a 'phi_2' parameter."
        assert "omega_1" in self.model.params, "Vaccinated: model params needs to have a 'omega_1' parameter."
        assert "omega_2" in self.model.params, "Vaccinated: model params needs to have a 'omega_2' parameter."
        assert "nu_1_jt" in self.model.params, "Vaccinated: model params needs to have a 'nu_1_jt' parameter."
        assert "nu_2_jt" in self.model.params, "Vaccinated: model params needs to have a 'nu_2_jt' parameter."

        assert "d_jt" in self.model.params, "Susceptible: model.params needs to have a 'd_jt' attribute."

        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        V1 = model.people.V1[tick]
        V1_next = model.people.V1[tick + 1]
        V2 = model.people.V2[tick]
        V2_next = model.people.V2[tick + 1]
        S_next = model.people.S[tick + 1]

        # propagate the current values forward (V(t+1) = V(t) + ∆V)
        V1_next[:] = V1
        V2_next[:] = V2

        # -natural mortality
        non_disease_deaths = model.prng.binomial(V1_next, -np.expm1(-model.params.d_jt[tick])).astype(V1_next.dtype)
        V1_next -= non_disease_deaths
        ndd_next = model.patches.non_disease_deaths[tick]
        ndd_next += non_disease_deaths

        non_disease_deaths = model.prng.binomial(V2_next, -np.expm1(-model.params.d_jt[tick])).astype(V2_next.dtype)
        V2_next -= non_disease_deaths
        ndd_next += non_disease_deaths

        # -waning immunity
        waned = model.prng.binomial(V1_next, -np.expm1(-model.params.omega_1)).astype(V1_next.dtype)
        V1_next -= waned
        S_next += waned  # waned return to Susceptible

        waned = model.prng.binomial(V2_next, -np.expm1(-model.params.omega_2)).astype(V2_next.dtype)
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

    def plot(self, fig: Optional[Figure] = None):  # pragma: no cover
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
