"""Infectious compartment — symptomatic and asymptomatic cholera cases, plus disease mortality.

Owns four state vectors:

- `model.people.Isym` — symptomatic infectious population.
- `model.people.Iasym` — asymptomatic infectious population.
- `model.patches.disease_deaths` — per-tick deaths from disease.
- `model.patches.new_symptomatic` — per-tick incident symptomatic cases.
- `model.patches.reported_cases` / `reported_deaths` — observed counts
  (subject to under-reporting via `rho`, `rho_deaths`, and the
  `chi_endemic` / `chi_epidemic` healthcare-access modifier).

On each tick the component:

1. Applies non-disease mortality (`d_jt`) to both `Isym` and `Iasym`.
2. Computes a per-patch `mu_jt` disease-mortality rate that blends a
   baseline, a tick-linear slope, and an epidemic-factor multiplier
   gated on whether `Isym / N` exceeded `epidemic_threshold` at the
   reporting-lag tick.
3. Samples disease deaths and recoveries (`gamma_1` for symptomatic,
   `gamma_2` for asymptomatic).
4. Progresses individuals out of `E` according to `iota`, splitting
   them into symptomatic (`sigma`) and asymptomatic (`1 - sigma`).
5. Updates the lagged `reported_cases` and `reported_deaths` series.

Force-of-infection — both human-to-human and environment-to-human — is
handled by the dedicated components downstream
([`HumanToHuman`][laser.cholera.metapop.humantohuman.HumanToHuman],
[`EnvToHuman`][laser.cholera.metapop.envtohuman.EnvToHuman]); this
component focuses on the within-compartment dynamics.
"""

from collections.abc import Iterator
from typing import TYPE_CHECKING
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
from laser.cholera.metapop.utils import check_attr
from laser.cholera.metapop.utils import check_key


class Infectious:
    """Infectious compartment: tracks `Isym(t)`, `Iasym(t)`, disease deaths, and reported cases.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Allocate `Isym` / `Iasym` and the disease-deaths / reporting bookkeeping vectors.

        Splits `params.I_j_initial` into symptomatic and asymptomatic
        seed populations using the symptomatic fraction `params.sigma`.

        Args:
            model: The `Model` instance. Must have `people`, `patches`,
                and `params` set up, with `params.I_j_initial` and
                `params.sigma` populated.

        Raises:
            AttributeError: When `model` is missing `people`, `patches`,
                or `params`.
            ValueError: When a required `params` key is missing (e.g.
                `I_j_initial`, `sigma`).
        """
        self.model = model

        check_attr(model, "people", "Infectious: model needs to have a 'people' attribute.")
        model.people.add_vector_property("Isym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.people.add_vector_property("Iasym", length=model.params.nticks + 1, dtype=np.int32, default=0)
        check_attr(model, "patches", "Infectious: model needs to have a 'patches' attribute.")
        model.patches.add_vector_property("disease_deaths", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("new_symptomatic", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("reported_cases", length=model.params.nticks + 1, dtype=np.int32, default=0)
        model.patches.add_vector_property("reported_deaths", length=model.params.nticks + 1, dtype=np.int32, default=0)
        check_attr(model, "params", "Infectious: model needs to have a 'params' attribute.")
        check_key(model.params, "I_j_initial", "Infectious: model params needs to have a 'I_j_initial' (initial infectious population) parameter.")
        check_key(self.model.params, "sigma", "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter.")
        model.people.Isym[0] = np.round(model.params.sigma * model.params.I_j_initial).astype(model.people.Isym.dtype)
        model.people.Iasym[0] = model.params.I_j_initial - model.people.Isym[0]

        return

    def check(self):
        """Validate every parameter and prerequisite attribute used by `__call__`.

        Asserts that `model.people.R` exists (so recovery can route into
        it) and that `params` contains `d_jt`, `mu_j_baseline`,
        `mu_j_slope`, `mu_j_epidemic_factor`, `epidemic_threshold`,
        `gamma_1`, `gamma_2`, `iota`, `sigma`, `rho`, and `rho_deaths`.

        Raises:
            AttributeError: When `model.people.R` is missing.
            ValueError: When any of the required `params` keys is missing
                (`d_jt`, `mu_j_baseline`, `mu_j_slope`,
                `mu_j_epidemic_factor`, `epidemic_threshold`, `gamma_1`,
                `gamma_2`, `iota`, `sigma`, `rho`, `rho_deaths`).
        """
        check_attr(self.model.people, "R", "Infectious: model.people needs to have a 'R' attribute.")
        check_key(self.model.params, "d_jt", "Infectious: model params needs to have a 'd_jt' (mortality rate) parameter.")

        check_key(
            self.model.params,
            "mu_j_baseline",
            "Infectious: model params needs to have a 'mu_j_baseline' (baseline disease mortality rate) parameter.",
        )
        check_key(self.model.params, "mu_j_slope", "Infectious: model params needs to have a 'mu_j_slope' (disease mortality rate slope) parameter.")
        check_key(
            self.model.params,
            "mu_j_epidemic_factor",
            "Infectious: model params needs to have a 'mu_j_epidemic_factor' (disease mortality rate epidemic factor) parameter.",
        )
        check_key(
            self.model.params,
            "epidemic_threshold",
            "Infectious: model params needs to have a 'epidemic_threshold' (disease mortality rate epidemic threshold) parameter.",
        )

        check_key(self.model.params, "gamma_1", "Infectious: model params needs to have a 'gamma_1' (recovery rate) parameter.")
        check_key(self.model.params, "gamma_2", "Infectious: model params needs to have a 'gamma_2' (recovery rate) parameter.")
        check_key(self.model.params, "iota", "Infectious: model params needs to have a 'iota' (progression rate) parameter.")
        check_key(self.model.params, "sigma", "Infectious: model params needs to have a 'sigma' (symptomatic fraction) parameter.")
        check_key(self.model.params, "rho", "Infectious: model params needs to have a 'rho' (detected/expected cases) parameter.")
        check_key(self.model.params, "rho_deaths", "Infectious: model params needs to have a 'rho_deaths' (detected/expected deaths) parameter.")
        if not hasattr(self.model.patches, "non_disease_deaths"):
            self.model.patches.add_vector_property("non_disease_deaths", length=self.model.params.nticks + 1, dtype=np.int32, default=0)
        # PERF: install `model.patches.non_disease_death_prob_jt` (the
        # `1 - exp(-d_jt)` cache) idempotently if no earlier component has.
        # Lets each consumer be exercised in isolation; the cache is built
        # exactly once across the pipeline.
        if not hasattr(self.model.patches, "non_disease_death_prob_jt"):
            self.model.patches.add_vector_property("non_disease_death_prob_jt", length=self.model.params.d_jt.shape[0], dtype=np.float32, default=0.0)
            self.model.patches.non_disease_death_prob_jt[:] = -np.expm1(-self.model.params.d_jt)

        # PERF: cache `1 - exp(-rate)` for the three scalar rates that
        # otherwise get exponentiated every tick.
        self._iota_prob = -np.expm1(-self.model.params.iota)
        self._gamma_1_prob = -np.expm1(-self.model.params.gamma_1)
        self._gamma_2_prob = -np.expm1(-self.model.params.gamma_2)

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """Advance `Isym` / `Iasym` one tick: deaths, recoveries, then `E -> I` progression.

        For each patch and within each of `Isym` and `Iasym`:

        1. Carry forward; subtract non-disease deaths (`d_jt`).
        2. For `Isym` only: compute `mu_jt = mu_baseline * (1 + slope *
            t_factor) * (1 + epidemic_factor * epidemic_flag)`, sample
            disease deaths from `Binomial(Isym, 1 - exp(-mu_jt))`, and
            update `patches.disease_deaths` and lagged `reported_deaths`.
        3. Sample recoveries (`gamma_1` for symptomatic, `gamma_2` for
            asymptomatic) and add to `R[tick + 1]`.
        4. Sample `E -> I` progression from `Binomial(E_next, 1 -
            exp(-iota))`, split by `sigma`, and add into `Isym` / `Iasym`.
        5. Update lagged `reported_cases` using `rho` and the
            `chi_endemic` / `chi_epidemic` healthcare-access modifier.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.

        Raises:
            AssertionError: When any compartment goes negative after a
                draw (indicates the implementation's invariants have
                regressed).
        """
        # Symptomatic
        Isym = model.people.Isym[tick]
        Is_next = model.people.Isym[tick + 1]
        Is_next[:] = Isym

        ## natural deaths (d_jt)
        # PERF: pre-computed `-np.expm1(-d_jt)` cached as model.patches.non_disease_death_prob_jt.
        # non_disease_deaths = model.prng.binomial(Is_next, -np.expm1(-model.params.d_jt[tick])).astype(Is_next.dtype)
        non_disease_deaths = model.prng.binomial(Is_next, model.patches.non_disease_death_prob_jt[tick]).astype(Is_next.dtype)
        Is_next -= non_disease_deaths
        ndd_next = model.patches.non_disease_deaths[tick]
        ndd_next += non_disease_deaths
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        ## disease deaths (mu)

        t_factor = tick / model.params.nticks  # 0 <= t_factor <= 1.0
        N = model.people.S[tick] + model.people.E[tick] + model.people.Isym[tick] + model.people.Iasym[tick] + model.people.R[tick]
        if hasattr(model.people, "V1"):
            N += model.people.V1[tick]
        if hasattr(model.people, "V2"):
            N += model.people.V2[tick]
        # Don't include V1inf or V2inf above, they're not "real" - just bookkeeping
        if (treport := int(tick - model.params.delta_reporting_cases)) >= 0:
            Ireported = model.people.Isym[treport]
            epidemic_flag = (Ireported > (model.params.epidemic_threshold * N)).astype(np.int32)
        else:
            epidemic_flag = np.zeros_like(N, dtype=np.int32)
        mu_jt = model.params.mu_j_baseline * (1 + model.params.mu_j_slope * t_factor) * (1 + model.params.mu_j_epidemic_factor * epidemic_flag)

        disease_deaths = model.prng.binomial(Is_next, -np.expm1(-mu_jt)).astype(Is_next.dtype)
        model.patches.disease_deaths[tick] = disease_deaths
        Is_next -= disease_deaths
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        idx_death_report = int(tick - model.params.delta_reporting_deaths)
        if idx_death_report >= 0:
            model.patches.reported_deaths[tick] += model.prng.binomial(
                model.patches.disease_deaths[idx_death_report], model.params.rho_deaths
            ).astype(model.patches.reported_deaths.dtype)

        ## recovery (gamma)
        # PERF: pre-computed `-np.expm1(-gamma_1)` cached as self._gamma_1_prob.
        # recovered = model.prng.binomial(Is_next, -np.expm1(-model.params.gamma_1)).astype(Is_next.dtype)
        recovered = model.prng.binomial(Is_next, self._gamma_1_prob).astype(Is_next.dtype)
        Is_next -= recovered
        R_next = model.people.R[tick + 1]
        R_next += recovered
        assert np.all(Is_next >= 0), f"Is_next should not go negative ({tick=}\n\t{Is_next=})"

        # Asymptomatic
        Iasym = model.people.Iasym[tick]
        Ia_next = model.people.Iasym[tick + 1]
        Ia_next[:] = Iasym

        ## natural deaths (d_jt)
        # PERF: pre-computed `-np.expm1(-d_jt)` cached as model.patches.non_disease_death_prob_jt.
        # non_disease_deaths = model.prng.binomial(Ia_next, -np.expm1(-model.params.d_jt[tick])).astype(Ia_next.dtype)
        non_disease_deaths = model.prng.binomial(Ia_next, model.patches.non_disease_death_prob_jt[tick]).astype(Ia_next.dtype)
        Ia_next -= non_disease_deaths
        ndd_next += non_disease_deaths
        assert np.all(Ia_next >= 0), f"Ia_next should not go negative ({tick=}\n\t{Ia_next=})"

        ## recovery
        # PERF: pre-computed `-np.expm1(-gamma_2)` cached as self._gamma_2_prob.
        # recovered = model.prng.binomial(Ia_next, -np.expm1(-model.params.gamma_2)).astype(Ia_next.dtype)
        recovered = model.prng.binomial(Ia_next, self._gamma_2_prob).astype(Ia_next.dtype)
        Ia_next -= recovered
        # R_next = model.people.R[tick + 1]
        R_next += recovered
        assert np.all(Ia_next >= 0), f"Ia_next should not go negative ({tick=}\n\t{Ia_next=})"

        # Use E_next here, can't progress deceased individuals
        E_next = model.people.E[tick + 1]
        # PERF: pre-computed `-np.expm1(-iota)` cached as self._iota_prob.
        # progressing = model.prng.binomial(E_next, -np.expm1(-model.params.iota)).astype(E_next.dtype)
        progressing = model.prng.binomial(E_next, self._iota_prob).astype(E_next.dtype)
        E_next -= progressing
        assert np.all(E_next >= 0), f"E_next should not go negative ({tick=}\n\t{E_next=})"

        ## new symptomatic infections
        new_symptomatic = np.round(model.params.sigma * progressing).astype(Is_next.dtype)
        new_asymptomatic = progressing - new_symptomatic
        Is_next += new_symptomatic
        Ia_next += new_asymptomatic
        model.patches.new_symptomatic[tick + 1] = new_symptomatic

        # Update reported cases
        idx_probe = tick - model.params.delta_reporting_cases
        if idx_probe >= 0:
            infected_fraction = model.people.Isym[idx_probe] / model.patches.N[idx_probe]
            # Use chi_endemic or chi_epidemic depending on local infected fraction.
            chi_eff = np.where(infected_fraction < model.params.epidemic_threshold, model.params.chi_endemic, model.params.chi_epidemic)
            model.patches.reported_cases[tick + 1] += np.round(
                model.prng.binomial(model.patches.new_symptomatic[idx_probe], model.params.rho) / chi_eff
            ).astype(model.patches.reported_cases.dtype)

        # human-to-human infection in humantohuman.py
        # environmental infection in envtohuman.py
        # recovery from infection in recovered.py

        return

    def plot(self, fig: Optional[Figure] = None) -> Iterator[str]:  # pragma: no cover
        """Yield four Matplotlib figures: symptomatic, asymptomatic, total, and reported-vs-actual cases.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            Four labels in order: `"Infectious (Symptomatic)"`,
            `"Infectious (Asymptomatic)"`, `"Infectious (Total)"`,
            `"Reported Cases"`.
        """
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
