from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Infectious:
    def __init__(self, model) -> None:
        self.model = model

        assert hasattr(model, "people"), "Infectious: model needs to have a 'people' attribute."
        model.people.add_vector_property("Isym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("Iasym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "patches"), "Infectious: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("expected_cases", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("disease_deaths", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("new_symptomatic", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("reported_cases", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "params"), "Infectious: model needs to have a 'params' attribute."
        assert "I_j_initial" in model.params, "Infectious: model params needs to have a 'I_j_initial' (initial infectious population) parameter."
        assert "sigma" in self.model.params, "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter."
        model.people.Isym[0] = np.round(model.params.sigma * model.params.I_j_initial).astype(model.people.Isym.dtype)
        model.people.Iasym[0] = model.params.I_j_initial - model.people.Isym[0]

        return

    def check(self):
        assert hasattr(self.model.people, "R"), "Infectious: model.people needs to have a 'S' attribute."
        assert "d_jt" in self.model.params, "Infectious: model params needs to have a 'd_jt' (mortality rate) parameter."

        assert "mu_j_baseline" in self.model.params, (
            "Infectious: model params needs to have a 'mu_j_baseline' (baseline disease mortality rate) parameter."
        )
        assert "mu_j_slope" in self.model.params, "Infectious: model params needs to have a 'mu_j_slope' (disease mortality rate slope) parameter."
        assert "mu_j_epidemic_factor" in self.model.params, (
            "Infectious: model params needs to have a 'mu_j_epidemic_factor' (disease mortality rate epidemic factor) parameter."
        )
        assert "epidemic_threshold" in self.model.params, (
            "Infectious: model params needs to have a 'epidemic_threshold' (disease mortality rate epidemic threshold) parameter."
        )

        assert "gamma_1" in self.model.params, "Infectious: model params needs to have a 'gamma_1' (recovery rate) parameter."
        assert "gamma_2" in self.model.params, "Infectious: model params needs to have a 'gamma_2' (recovery rate) parameter."
        assert "iota" in self.model.params, "Infectious: model params needs to have a 'iota' (progression rate) parameter."
        assert "sigma" in self.model.params, "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter."
        assert "rho" in self.model.params, "Infectious: model params needs to have a 'rho' (detected/expected cases) parameter."
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)

        return

    def __call__(self, model, tick: int) -> None:
        # Symptomatic
        Isym = model.people.Isym[tick]
        Is_next = model.people.Isym[tick + 1]
        Is_next[:] = Isym

        ## natural deaths (d_jt)
        non_disease_deaths = model.prng.binomial(Is_next, -np.expm1(-model.params.d_jt[tick])).astype(Is_next.dtype)
        Is_next -= non_disease_deaths
        ndd_next = model.patches.non_disease_deaths[tick]
        ndd_next += non_disease_deaths
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        ## disease deaths (mu)

        t_factor = tick / model.params.nticks
        N = model.people.S[tick] + model.people.E[tick] + model.people.Isym[tick] + model.people.Iasym[tick] + model.people.R[tick]
        if hasattr(model.people, "V1imm"):
            N += model.people.V1imm[tick] + model.people.V1sus[tick] + model.people.V2imm[tick] + model.people.V2sus[tick]
        # Don't include V1inf or V2inf above, they're not "real" but just bookkeeping
        if tick >= (delta := model.params.delta_reporting_cases):
            treport = int(tick - delta)
            Ireported = model.people.Isym[treport]
            epidemic_flag = (Ireported > (model.params.epidemic_threshold * N)).astype(np.int32)
        else:
            epidemic_flag = np.zeros_like(N).astype(np.float32)
        mu_jt = model.params.mu_j_baseline * (1 + model.params.mu_j_slope * t_factor) * (1 + model.params.mu_j_epidemic_factor * epidemic_flag)

        disease_deaths = model.prng.binomial(Is_next, -np.expm1(-mu_jt)).astype(Is_next.dtype)
        model.patches.disease_deaths[tick] = disease_deaths
        Is_next -= disease_deaths
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        ## recovery (gamma)
        recovered = model.prng.binomial(Is_next, -np.expm1(-model.params.gamma_1)).astype(Is_next.dtype)
        Is_next -= recovered
        R_next = model.people.R[tick + 1]
        R_next += recovered
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        # Asymptomatic
        Iasym = model.people.Iasym[tick]
        Ia_next = model.people.Iasym[tick + 1]
        Ia_next[:] = Iasym

        ## natural deaths (d_jt)
        non_disease_deaths = model.prng.binomial(Ia_next, -np.expm1(-model.params.d_jt[tick])).astype(Ia_next.dtype)
        Ia_next -= non_disease_deaths
        ndd_next += non_disease_deaths
        assert np.all(Ia_next >= 0), f"Ia_next should not go negative ({tick=}\n\t{Ia_next=})"

        ## recovery
        recovered = model.prng.binomial(Ia_next, -np.expm1(-model.params.gamma_2)).astype(Ia_next.dtype)
        Ia_next -= recovered
        # R_next = model.people.R[tick + 1]
        R_next += recovered
        assert np.all(Ia_next >= 0), f"Ia_next should not go negative ({tick=}\n\t{Ia_next=})"

        # Use E_next here, can't progress deceased individuals
        E_next = model.people.E[tick + 1]
        progressing = model.prng.binomial(E_next, -np.expm1(-model.params.iota)).astype(E_next.dtype)
        E_next -= progressing
        assert np.all(E_next >= 0), f"E_next should not go negative ({tick=}\n\t{E_next=})"

        ## new symptomatic infections
        new_symptomatic = np.round(model.params.sigma * progressing).astype(Is_next.dtype)
        new_asymptomatic = progressing - new_symptomatic
        Is_next += new_symptomatic
        Ia_next += new_asymptomatic
        model.patches.new_symptomatic[tick + 1] = new_symptomatic

        # Update expected cases
        expected_cases = model.patches.expected_cases[tick + 1]
        expected_cases += np.round(new_symptomatic / model.params.rho).astype(expected_cases.dtype)

        # Update reported cases
        idx_probe = tick - model.params.delta_reporting_cases
        if idx_probe >= 0:
            infected_fraction = model.people.Isym[idx_probe] / model.patches.N[idx_probe]
            # Use chi_endemic or chi_epidemic depending on local infected fraction.
            chi_eff = np.where(infected_fraction < model.params.epidemic_threshold, model.params.chi_endemic, model.params.chi_epidemic)
            # Note that sigma is already factored into the calculation of Isym (see above).
            model.patches.reported_cases[tick + 1] += np.round(model.people.Isym[idx_probe] * model.params.rho / chi_eff).astype(
                model.patches.reported_cases.dtype
            )

        # human-to-human infection in humantohuman.py
        # environmental infection in envtohuman.py
        # recovery from infection in recovered.py

        return

    def plot(self, fig: Optional[Figure] = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Infectious (Symptomatic)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.Isym[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Symptomatic")
        plt.legend()

        yield "Infectious (Symptomatic)"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Infectious (Asymptomatic)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.Iasym[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Asymptomatic")
        plt.legend()

        yield "Infectious (Asymptomatic)"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Infectious (Total)") if fig is None else fig

        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.people.Isym[:, ipatch] + self.model.people.Iasym[:, ipatch], label=f"{self.model.params.location_name[ipatch]}")
        plt.xlabel("Tick")
        plt.ylabel("Total Infectious")
        plt.legend()

        yield "Infectious (Total)"

        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Reported vs. Actual Cases") if fig is None else fig

        plt.plot(self.model.patches.reported_cases.sum(axis=1), color="blue", label="Reported")
        plt.plot(self.model.people.Isym.sum(axis=1), color="red", label="Actual")
        plt.xlabel("Tick")
        plt.ylabel("Cases")
        plt.legend()

        yield "Reported Cases"
        return
