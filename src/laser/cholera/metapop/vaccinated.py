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

        # +newly vaccinated (successful take)
        new_first_doses_delivered = np.round(model.params.nu_1_jt[tick]).astype(V1.dtype)
        if np.any(new_first_doses_delivered > S_next):
            logger.debug(f"WARNING: new_first_doses_delivered > S_next ({tick=})")
            for index in np.nonzero(new_first_doses_delivered > S_next)[0]:
                logger.debug(f"\t{model.params.location_name[index]}: doses {new_first_doses_delivered[index]} > {S_next[index]} susceptible")
            new_first_doses_delivered = np.minimum(new_first_doses_delivered, S_next)
        model.patches.dose_one_doses[tick] = new_first_doses_delivered
        assert np.all(S_next >= 0), f"'S' should not go negative ({tick=}\n\t{S_next})"
        # effective doses
        newly_immunized = np.round(model.params.phi_1 * new_first_doses_delivered).astype(V1_next.dtype)
        # just move the effective doses, leave ineffective doses in S
        S_next -= newly_immunized
        V1_next += newly_immunized

        # -second dose recipients
        new_second_doses_delivered = np.round(model.params.nu_2_jt[tick]).astype(V2_next.dtype)
        if np.any(new_second_doses_delivered > V1):
            logger.debug(f"WARNING: new_second_doses_delivered > V1 ({tick=}\n\t{new_second_doses_delivered=}\n\t{V1=})")
            new_second_doses_delivered = np.minimum(new_second_doses_delivered, V1)
        model.patches.dose_two_doses[tick] = new_second_doses_delivered

        # effective doses
        newly_immunized = np.round(model.params.phi_2 * new_second_doses_delivered).astype(V2_next.dtype)
        # just move the effective doses, leave ineffective doses in V1
        V1_next -= newly_immunized
        V2_next += newly_immunized

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
