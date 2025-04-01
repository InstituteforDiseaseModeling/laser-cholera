import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure


class Infectious:
    def __init__(self, model, verbose: bool = False) -> None:
        self.model = model
        self.verbose = verbose

        assert hasattr(model, "agents"), "Infectious: model needs to have a 'agents' attribute."
        model.agents.add_vector_property("Isym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.agents.add_vector_property("Iasym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "patches"), "Infectious: model needs to have a 'patches' attribute."
        model.patches.add_vector_property("expected_cases", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("disease_deaths", length=model.params.nticks + 1, dtype=np.int32, default=0)
        assert hasattr(model, "params"), "Infectious: model needs to have a 'params' attribute."
        assert hasattr(model.params, "I_j_initial"), (
            "Infectious: model params needs to have a 'I_j_initial' (initial infectious population) parameter."
        )
        assert "sigma" in self.model.params, "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter."
        model.agents.Isym[0] = np.round(model.params.sigma * model.params.I_j_initial).astype(model.agents.Isym.dtype)
        model.agents.Iasym[0] = model.params.I_j_initial - model.agents.Isym[0]

        return

    def check(self):
        assert hasattr(self.model.agents, "R"), "Infectious: model.agents needs to have a 'S' attribute."
        assert "d_jt" in self.model.params, "Infectious: model params needs to have a 'd_jt' (mortality rate) parameter."
        assert "mu_jt" in self.model.params, "Infectious: model params needs to have a 'mu_jt' (disease mortality rate) parameter."
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
        Isym = model.agents.Isym[tick]
        Is_next = model.agents.Isym[tick + 1]
        Is_next[:] = Isym

        ## natural deaths (d_jt)
        non_disease_deaths = model.prng.binomial(Is_next, -np.expm1(-model.params.d_jt[tick])).astype(Is_next.dtype)
        Is_next -= non_disease_deaths
        ndd_next = model.patches.non_disease_deaths[tick + 1]
        ndd_next += non_disease_deaths
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        ## disease deaths (mu)
        disease_deaths = model.prng.binomial(Is_next, -np.expm1(-model.params.mu_jt[tick])).astype(Is_next.dtype)
        model.patches.disease_deaths[tick] = disease_deaths
        Is_next -= disease_deaths
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        ## recovery (gamma)
        recovered = model.prng.binomial(Is_next, -np.expm1(-model.params.gamma_1)).astype(Is_next.dtype)
        Is_next -= recovered
        R_next = model.agents.R[tick + 1]
        R_next += recovered
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        # Asymptomatic
        Iasym = model.agents.Iasym[tick]
        Ia_next = model.agents.Iasym[tick + 1]
        Ia_next[:] = Iasym

        ## natural deaths (d_jt)
        non_disease_deaths = model.prng.binomial(Ia_next, -np.expm1(-model.params.d_jt[tick])).astype(Ia_next.dtype)
        Ia_next -= non_disease_deaths
        assert np.all(Ia_next >= 0), f"Ia_next should not go negative ({tick=}\n\t{Ia_next=})"

        ## recovery
        recovered = model.prng.binomial(Ia_next, -np.expm1(-model.params.gamma_2)).astype(Ia_next.dtype)
        Ia_next -= recovered
        # R_next = model.agents.R[tick + 1]
        R_next += recovered
        assert np.all(Ia_next >= 0), f"Ia_next should not go negative ({tick=}\n\t{Ia_next=})"

        # Use E_next here, can't progress deceased individuals
        E_next = model.agents.E[tick + 1]
        progress = model.prng.binomial(E_next, -np.expm1(-model.params.iota)).astype(E_next.dtype)
        E_next -= progress
        assert np.all(E_next >= 0), f"E_next should not go negative ({tick=}\n\t{E_next=})"

        ## new symptomatic infections
        new_symptomatic = np.round(model.params.sigma * progress).astype(Is_next.dtype)
        new_asymptomatic = progress - new_symptomatic
        Is_next += new_symptomatic
        Ia_next += new_asymptomatic

        # Update expected cases
        expected_cases = model.patches.expected_cases[tick + 1]
        expected_cases += np.round(new_symptomatic / model.params.rho).astype(expected_cases.dtype)

        # human-to-human infection in humantohuman.py
        # environmental infection in envtohuman.py
        # recovery from infection in recovered.py

        return

    def plot(self, fig: Figure = None):  # pragma: no cover
        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Infectious (Symptomatic)")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.Isym[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Infectious (Asymptomatic)")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.Iasym[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield

        _fig = plt.figure(figsize=(12, 9), dpi=128) if fig is None else fig

        plt.title("Infectious (Total)")
        for ipatch in np.argsort(self.model.params.S_j_initial)[-10:]:
            plt.plot(self.model.agents.Isym[:, ipatch] + self.model.agents.Iasym[:, ipatch], label=f"Patch {ipatch}")
        plt.xlabel("Tick")
        plt.legend()

        yield
        return
