import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)


class Vaccinated:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "people"), "Vaccinated: model needs to have a 'people' attribute."
        model.people.add_vector_property("V1imm", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V1sus", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V1inf", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V2imm", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V2sus", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("V2inf", length=model.params.nticks + 1, dtype=np.int32, default=0)
        # We will track doses on the date (tick) given to more easily match nu_1_jt and nu_2_jt.
        model.patches.add_vector_property("V1", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("V2", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("dose_one_doses", length=model.params.nticks, dtype=np.int32, default=0)
        model.patches.add_vector_property("dose_two_doses", length=model.params.nticks, dtype=np.int32, default=0)
        assert "V1_j_initial" in model.params, (
            "Vaccinated: model params needs to have a 'V1_j_initial' (initial one dose vaccinated population) parameter."
        )
        assert "V2_j_initial" in model.params, (
            "Vaccinated: model params needs to have a 'V2_j_initial' (initial two dose vaccinated population) parameter."
        )
        assert "phi_1" in model.params, "Vaccinated: model params needs to have a 'phi_1' (efficacy of one dose) parameter."
        assert "phi_2" in model.params, "Vaccinated: model params needs to have a 'phi_2' (efficacy of two doses) parameter."
        model.people.V1imm[0] = np.round(model.params.phi_1 * model.params.V1_j_initial)
        model.people.V1sus[0] = model.params.V1_j_initial - model.people.V1imm[0]
        model.people.V1inf[0] = 0
        model.patches.V1[0] = model.people.V1imm[0] + model.people.V1sus[0] + model.people.V1inf[0]
        model.people.V2imm[0] = np.round(model.params.phi_2 * model.params.V2_j_initial)
        model.people.V2sus[0] = model.params.V2_j_initial - model.people.V2imm[0]
        model.people.V2inf[0] = 0
        model.patches.V2[0] = model.people.V2imm[0] + model.people.V2sus[0] + model.people.V2inf[0]

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

        assert hasattr(self.model.params, "d_jt"), "Susceptible: model.params needs to have a 'd_jt' attribute."

        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        V1_next = model.patches.V1[tick + 1]
        V2_next = model.patches.V2[tick + 1]
        V1imm = model.people.V1imm[tick]
        V1sus = model.people.V1sus[tick]
        V1inf = model.people.V1inf[tick]
        V1imm_next = model.people.V1imm[tick + 1]
        V1sus_next = model.people.V1sus[tick + 1]
        V1inf_next = model.people.V1inf[tick + 1]
        V2imm = model.people.V2imm[tick]
        V2sus = model.people.V2sus[tick]
        V2inf = model.people.V2inf[tick]
        V2imm_next = model.people.V2imm[tick + 1]
        V2sus_next = model.people.V2sus[tick + 1]
        V2inf_next = model.people.V2inf[tick + 1]
        S = model.people.S[tick]
        E = model.people.E[tick]
        S_next = model.people.S[tick + 1]

        V1imm_next[:] = V1imm
        V1sus_next[:] = V1sus
        V1inf_next[:] = V1inf
        V2imm_next[:] = V2imm
        V2sus_next[:] = V2sus
        V2inf_next[:] = V2inf

        # -natural mortality
        non_disease_deaths = model.prng.binomial(V1imm, -np.expm1(-model.params.d_jt[tick])).astype(V1imm.dtype)
        V1imm_next -= non_disease_deaths
        ndd_next = model.patches.non_disease_deaths[tick]
        ndd_next += non_disease_deaths
        non_disease_deaths = model.prng.binomial(V1sus, -np.expm1(-model.params.d_jt[tick])).astype(V1sus.dtype)
        V1sus_next -= non_disease_deaths
        ndd_next += non_disease_deaths
        # this won't exactly match reality as the _actual_ people merged back into the E compartment upon infection
        non_disease_deaths = model.prng.binomial(V1inf, -np.expm1(-model.params.d_jt[tick])).astype(V1inf.dtype)
        V1inf_next -= non_disease_deaths
        # ndd_next += non_disease_deaths  # don't include here as V1inf is just a counter

        non_disease_deaths = model.prng.binomial(V2imm, -np.expm1(-model.params.d_jt[tick])).astype(V2imm.dtype)
        V2imm_next -= non_disease_deaths
        ndd_next += non_disease_deaths
        non_disease_deaths = model.prng.binomial(V2sus, -np.expm1(-model.params.d_jt[tick])).astype(V2sus.dtype)
        V2sus_next -= non_disease_deaths
        ndd_next += non_disease_deaths
        # this won't exactly match reality as the _actual_ people merged back into the E compartment upon infection
        non_disease_deaths = model.prng.binomial(V2inf, -np.expm1(-model.params.d_jt[tick])).astype(V2inf.dtype)
        V2inf_next -= non_disease_deaths
        # ndd_next += non_disease_deaths # don't include here as V2inf is just a counter

        # -waning immunity
        waned = model.prng.binomial(V1imm_next, -np.expm1(-model.params.omega_1)).astype(V1imm_next.dtype)
        V1imm_next -= waned
        V1sus_next += waned

        waned = model.prng.binomial(V2imm_next, -np.expm1(-model.params.omega_2)).astype(V2imm_next.dtype)
        V2imm_next -= waned
        V2sus_next += waned

        # +newly vaccinated (successful take)
        new_one_doses = model.prng.poisson(model.params.nu_1_jt[tick] * S / (S + E)).astype(V1imm.dtype)
        model.patches.dose_one_doses[tick] = new_one_doses
        if np.any(new_one_doses > S):
            logger.debug(f"WARNING: new_one_doses > S ({tick=})")
            for index in np.nonzero(new_one_doses > S)[0]:
                logger.debug(f"\t{model.params.location_name[index]}: doses {new_one_doses[index]} > {S[index]} susceptible")
            new_one_doses = np.minimum(new_one_doses, S)
        S_next -= new_one_doses
        assert np.all(S_next >= 0), f"S' should not go negative ({tick=}\n\t{S_next})"
        # effective doses
        new_immunized = np.round(model.params.phi_1 * new_one_doses).astype(V1imm.dtype)
        V1imm_next += new_immunized
        # infective doses
        new_ineffective = new_one_doses - new_immunized
        V1sus_next += new_ineffective

        # -second dose recipients
        # set minimum V1 to 1 to avoid division by zero
        V1 = np.maximum(V1imm + V1sus + V1inf, 1).astype(V1imm.dtype)
        new_two_doses = model.prng.poisson(model.params.nu_2_jt[tick]).astype(V1imm.dtype)
        if np.any(new_two_doses > V1):
            logger.debug(f"WARNING: new_two_doses > V1 ({tick=}\n\t{new_two_doses=}\n\t{V1=})")
            new_two_doses = np.minimum(new_two_doses, V1)
        model.patches.dose_two_doses[tick] = new_two_doses
        v1imm_contribution = np.round((V1imm / V1) * new_two_doses).astype(V1imm.dtype)
        V1imm_next -= v1imm_contribution
        v1sus_contribution = np.round((V1sus / V1) * new_two_doses).astype(V1sus.dtype)
        V1sus_next -= v1sus_contribution
        v1inf_contribution = np.round((new_two_doses - v1imm_contribution - v1sus_contribution) * (V1inf / V1)).astype(V1inf.dtype)
        V1inf_next -= v1inf_contribution

        assert np.all(V1imm_next >= 0), f"V1imm' should not go negative ({tick=}\n\t{V1imm_next})"
        assert np.all(V1sus_next >= 0), f"V1sus' should not go negative ({tick=}\n\t{V1sus_next})"
        assert np.all(V1inf_next >= 0), f"V1inf' should not go negative ({tick=}\n\t{V1inf_next})"

        # effective doses
        # TODO - use new_two_doses here or (v1imm_contribution + v1sus_contribution)?
        new_immunized = np.round(model.params.phi_2 * ((V1imm + V1sus) / V1) * new_two_doses).astype(V2imm.dtype)
        V2imm_next += new_immunized
        # infective doses
        new_infective = np.round((1 - model.params.phi_2) * ((V1imm + V1sus) / V1) * new_two_doses).astype(V2sus.dtype)
        V2sus_next += new_infective
        # doses applied to previously infected one dose recipients
        v2inf_delta = new_two_doses - new_immunized - new_infective
        if np.any(v2inf_delta < 0):
            logger.debug(f"WARNING: v2inf_delta < 0 ({tick=}\n\t{v2inf_delta=})")
            v2inf_delta = np.maximum(v2inf_delta, 0)
        V2inf_next += v2inf_delta

        # V1 total
        V1_next[:] = V1imm_next + V1sus_next + V1inf_next

        # V2 total
        V2_next[:] = V2imm_next + V2sus_next + V2inf_next

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Vaccinated (One Dose)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.V1[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Vaccinated (One Dose)")
        plt.legend()

        yield "Vaccinated (One Dose)"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Vaccinated (Two Doses)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.patches.V2[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Vaccinated (Two Doses)")
        plt.legend()

        yield "Vaccinated (Two Doses)"
        return
