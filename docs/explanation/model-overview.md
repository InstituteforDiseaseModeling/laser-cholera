# Model overview

This page is the central Explanation for `laser.cholera`: it sketches what
kind of model the package implements, what the simulation state actually
*is* (the compartments and the per-patch derived quantities), and how a
single tick is assembled out of an ordered list of components that each
own one slice of that state. It assumes you have skimmed the
[parameter reference](../reference/parameters/index.md) and the
[first-run tutorial](../tutorials/first-run.md), but it does not require
prior exposure to the upstream MOSAIC literature. After reading you
should be able to look at any per-tick component in `src/laser/cholera/
metapop/` and place it inside the wider pipeline, and you should be able
to predict — given a parameter file — roughly which arrays will be
populated and which will stay at their defaults.

## What kind of model is this?

`laser.cholera` is a **stochastic compartmental SEIRV metapopulation
model**, integrated forward in daily ticks. Three words in that
description do the heavy lifting.

*Stochastic.* Every flow between compartments is sampled. Births are
Poisson draws (`Poisson(N * b_jt)`); non-disease deaths, disease deaths,
recoveries, vaccinations, and new infections are all Binomial draws of
the form `Binomial(state, 1 - exp(-rate))`; environmental shedding and
decay are Poisson. The simulation is fully reproducible from a single
integer seed (`params.seed`, threaded into `seed_prng(...)` in
`Model.__init__`); the bit-for-bit baseline contract — checked by
`misc/perf_baseline.py` — depends on this.

*Compartmental.* Each individual is in one of seven mutually exclusive
states (S, E, Isym, Iasym, R, V1, V2; see below). The model does not
track agents; it tracks the *count* of people in each compartment in
each patch on each day. Internally those counts live on
`model.people` — a `LaserFrame` of length `npatches`, with one
`int32(nticks + 1, npatches)` array per compartment.

*Metapopulation.* The simulation runs `npatches` parallel
mini-populations and couples them through a fixed gravity-style mobility
matrix `pi_ij`. Patches are whatever the user provides: countries
(the bundled SSA baseline), admin-1 units, admin-2 units, or even a
single patch (the toy configuration). The mathematics is identical at
every resolution; only the JSON inputs change shape.

The temporal grid is days. `params.nticks` is the number of daily
updates; `model.people.S` (and every other compartment array) has shape
`(nticks + 1, npatches)` because index `0` stores the initial state
(seeded from the `*_j_initial` parameters in each component's
`__init__`) and indices `1 … nticks` store the state after each
successive tick.

## The seven compartments

Every entry below is allocated by exactly one component during `Model`
construction and updated every tick. The seeded value at `t = 0` comes
from the matching `*_j_initial` parameter.

| Symbol | Compartment | Allocated by |
|---|---|---|
| `S` | Susceptible — never infected and not currently vaccine-protected. | `Susceptible.__init__` |
| `E` | Exposed — infected but not yet infectious (incubating). | `Exposed.__init__` |
| `Isym` | Symptomatic infectious — currently shedding *and* clinically apparent. | `Infectious.__init__` |
| `Iasym` | Asymptomatic infectious — currently shedding but clinically inapparent. | `Infectious.__init__` |
| `R` | Recovered — protected by natural immunity; wanes back to `S` at rate `epsilon`. | `Recovered.__init__` |
| `V1` | One-dose vaccinated — protected by a single dose; wanes back to `S` at rate `omega_1`. | `Vaccinated.__init__` |
| `V2` | Two-dose vaccinated — protected by two doses; wanes back to `S` at rate `omega_2`. | `Vaccinated.__init__` |

The `Isym` / `Iasym` split is governed by a single scalar `sigma`
(symptomatic fraction): when `E → I` progression fires inside
`Infectious.__call__`, the progressing count is split `sigma : 1 - sigma`
between the two compartments. The two flavours of infectious shed into
the environmental reservoir at independent rates (`zeta_1`, `zeta_2`)
and recover at independent rates (`gamma_1`, `gamma_2`), but otherwise
behave identically.

Mathematically the per-patch flow looks like:

$$
S \xrightarrow{\Lambda + \Psi} E
\xrightarrow{\iota}
\begin{cases} I_{\text{sym}} & \text{prob. } \sigma \\ I_{\text{asym}} & \text{prob. } 1 - \sigma \end{cases}
\xrightarrow{\gamma_1,\gamma_2} R \xrightarrow{\varepsilon} S
$$

with vaccination flows `S, E, Isym, Iasym, R → V1 → V2 → S` driven by
`nu_1_jt`, `nu_2_jt`, `phi_1`, `phi_2`, `omega_1`, `omega_2`, and births
and natural deaths layered on top of every compartment via `b_jt` and
`d_jt`.

## Per-patch derived quantities

In addition to the seven compartments, the components allocate a
collection of per-patch series on `model.patches`. These are the inputs
and outputs that link the compartments to one another and to the
parameter set. They group naturally by who owns them.

**Population bookkeeping** (`Census`, `Susceptible`, `Infectious`)

- `N` — total per-patch population (sum of the seven compartments).
- `births` — Poisson-sampled births at rate `N * b_jt`.
- `non_disease_deaths` — Binomial-sampled deaths at rate `d_jt`,
  aggregated across all alive compartments.
- `disease_deaths` — Binomial-sampled deaths-from-disease at rate
  `mu_jt`, from `Isym` only.
- `non_disease_death_prob_jt` — performance cache of `1 - exp(-d_jt)`,
  shared across every component that draws on it.

**Force-of-infection** (`HumanToHuman`, `EnvToHuman`, `Environmental`)

- `Lambda` — per-tick human-to-human force-of-infection (the hazard a
  susceptible person experiences from infectious neighbours).
- `Psi` — per-tick environmental force-of-infection (the hazard from
  the contaminated-water reservoir).
- `W` — environmental reservoir of cholera; accumulated from shedding,
  drained by decay (`delta_jt`).
- `beta_jt_human` — seasonality envelope for human-to-human
  transmission, baked once at construction time from `a_*_j`, `b_*_j`,
  `p` and `beta_j0_hum`.
- `beta_jt_env` — seasonality envelope for environmental transmission,
  baked once at construction time from `psi_jt` and `beta_j0_env`.
- `delta_jt` — per-tick environmental decay rate, derived once from
  `psi_jt` via the Beta-CDF map in `Environmental`.

**Incidence and reporting** (`Infectious`, `EnvToHuman`, `HumanToHuman`)

- `incidence_human` — per-tick new infections caused by direct
  transmission.
- `incidence_env` — per-tick new infections caused by environmental
  transmission.
- `incidence` — per-tick total new infections (the sum of the two
  splits).
- `new_symptomatic` — per-tick incident symptomatic cases (after the
  `sigma` split).
- `reported_cases` — Binomial-thinned observed cases, lagged by
  `delta_reporting_cases` ticks and modulated by `rho` and the
  `chi_endemic` / `chi_epidemic` regime modifier.
- `reported_deaths` — Binomial-thinned observed deaths, lagged by
  `delta_reporting_deaths` ticks and modulated by `rho_deaths`.

**Vaccination bookkeeping** (`Vaccinated`)

- `dose_one_doses` / `dose_two_doses` — per-tick dose counts, indexed
  to the day delivered for direct alignment with `nu_1_jt` / `nu_2_jt`.

**Coupling and diagnostics** (`HumanToHuman`, `DerivedValues`)

- `pi_ij` — gravity-model mobility matrix derived once from
  `latitude`, `longitude`, `mobility_omega`, and `mobility_gamma`. Even
  with mobility off (`tau_i = 0`) the matrix is still computed; it just
  has no effect on the dynamics.
- `coupling` — `(npatches, npatches)` Pearson correlation matrix of
  per-tick prevalence-fraction series, computed once on the final tick
  by `DerivedValues`.
- `spatial_hazard` — per-tick, per-patch infection-pressure summary,
  also computed by `DerivedValues` on the final tick.

The `RInterface` view on `model.results` re-exposes most of these
arrays trimmed of the `t = 0` seed row and transposed to
`[npatches, nticks]`, so R consumers see the layout their reference
implementation expects.

## The per-tick component pipeline

The default pipeline is wired in `run_model` (`model.py`) as a list of
thirteen component classes:

```python
model.components = [
    Susceptible,
    Exposed,
    Recovered,
    Infectious,
    Vaccinated,
    Census,
    HumanToHuman,
    EnvToHuman,
    Environmental,
    DerivedValues,
    Analyzer,
    Recorder,
    Parameters,
]
```

Setting `model.components = [...]` does three things. First, each class
is instantiated against the model — its `__init__` allocates the state
arrays it owns. Second, each instance with a `__call__` method is
appended to `model.phases` (the run loop's call list). Third, every
instance's `check()` method is invoked once, before the first tick, to
validate that any state allocated by *other* components is in place
(e.g. `Infectious.check()` asserts `model.people.R` exists, even though
`Recovered` is what allocated it).

Inside `Model.run`, each tick walks `self.phases` in list order and
calls `phase(self, tick)`. Components mutate `model.people` and
`model.patches` in place; the canonical execution order matters
because each component reads slices that the previous one wrote.

| # | Component | What it does each tick |
|---|---|---|
| 1 | `Susceptible` | Carries `S` forward; subtracts non-disease deaths; adds births. |
| 2 | `Exposed` | Carries `E` forward; subtracts non-disease deaths. (`E → I` progression happens inside `Infectious`.) |
| 3 | `Recovered` | Carries `R` forward; subtracts non-disease deaths; wanes a fraction `1 - exp(-epsilon)` back to `S`. |
| 4 | `Infectious` | Carries `Isym` / `Iasym` forward; subtracts non-disease deaths; samples `mu_jt`-driven disease deaths from `Isym`; samples `gamma_1` / `gamma_2` recoveries into `R`; progresses `E → I` at rate `iota` with `sigma` split; updates `reported_cases` / `reported_deaths` at the appropriate lag. |
| 5 | `Vaccinated` | Carries `V1` / `V2` forward; subtracts non-disease deaths; wanes `V1 → S` at `omega_1` and `V2 → S` at `omega_2`; delivers `nu_2_jt` second doses from `V1 → V2`; delivers `nu_1_jt` first doses from the source compartments (default `S, E, Isym, Iasym, R`) into `V1`. |
| 6 | `Census` | Sums `S + E + Isym + Iasym + R + V1 + V2` into `patches.N[tick + 1]`. Other downstream components read `N` rather than recomputing it. |
| 7 | `HumanToHuman` | Computes `Lambda` from `Isym + Iasym` weighted by `pi_ij` and `beta_jt_human`; samples new `S → E` infections; records into `incidence_human` and `incidence`. |
| 8 | `EnvToHuman` | Computes `Psi` from the reservoir `W` and the seasonality envelope `beta_jt_env`, attenuated by `(1 - theta_j)` (WASH coverage) and saturated by `W / (kappa + W)`; samples new `S → E` infections; records into `incidence_env` and `incidence`. |
| 9 | `Environmental` | Updates `W`: Poisson decay at rate `delta_jt * W`, Poisson shedding at `zeta_1 * Isym` and `zeta_2 * Iasym` attenuated by `(1 - theta_j)`. |
| 10 | `DerivedValues` | No-op until the final tick, when it computes `spatial_hazard` and the `coupling` correlation matrix. |
| 11 | `Analyzer` | No-op until the final tick, when (if `params.calc_likelihood` is set) it calls `calc_model_likelihood` and stashes the result on `model.log_likelihood`. |
| 12 | `Recorder` | No-op until the final tick, when (if `params.hdf5_output` is set) it writes the whitelisted properties to an HDF5 file. |
| 13 | `Parameters` | Per-tick no-op; contributes only to `model.visualize()` (renders nine summary plots of the input parameter set). |

The ordering is deliberate. The five compartment components (1–5) run
first so that every compartment has written `[tick + 1]` before
`Census` (6) totals them. `Census` then runs before the two
force-of-infection components (7, 8) because they read `N`. `EnvToHuman`
(8) runs *before* `Environmental` (9) because the reservoir `W[tick]`
that drives this tick's environmental hazard is the one written by the
*previous* tick's `Environmental`; if the order were reversed, the
hazard would see this tick's shedding immediately, collapsing the
intended one-tick lag. `DerivedValues` and `Analyzer` (10, 11) only do
real work on the last tick, when the full trajectories are available.
`Recorder` (12) and `Parameters` (13) are output-only.

## Parameter set → state mapping

Every parameter in `default_parameters.json` feeds exactly one (or a
small handful) of these components. The reference pages document each
parameter's shape, range, and off-value individually; the table below
just shows where to look when you want to know which knob controls
which array.

| Parameter group | Primary consumer(s) | Drives |
|---|---|---|
| Run identity (`seed`, `date_start`, `date_stop`, `location_name`) | `Model.__init__` | PRNG seed, `nticks`, `npatches` |
| Initial populations (`*_j_initial`, `prop_*_initial`) | All five compartment components | The `[t = 0]` slice of `S`, `E`, `Isym`, `Iasym`, `R`, `V1`, `V2` |
| Vital dynamics (`b_jt`, `d_jt`, `mu_j_*`, `mu_jt`) | `Susceptible`, `Infectious` (and `d_jt` is read by every compartment) | `births`, `non_disease_deaths`, `disease_deaths` |
| Vaccination (`nu_1_jt`, `nu_2_jt`, `phi_*`, `omega_*`, `nu_jt_sources`) | `Vaccinated` | `V1`, `V2`, `dose_one_doses`, `dose_two_doses` |
| Disease progression (`iota`, `gamma_1`, `gamma_2`, `epsilon`, `sigma`) | `Exposed`, `Infectious`, `Recovered` | `E → I`, `I → R`, `R → S`, `sigma` split |
| Reporting (`rho`, `rho_deaths`, `delta_reporting_*`, `reported_*`) | `Infectious`, `Analyzer` | `reported_cases`, `reported_deaths` |
| Regime switching (`chi_endemic`, `chi_epidemic`, `epidemic_threshold`) | `Infectious` | The `mu_jt` / reporting epidemic-vs-endemic branch |
| Geography & mobility (`latitude`, `longitude`, `tau_i`, `mobility_*`) | `HumanToHuman` | `pi_ij`, the contribution of neighbour patches to `Lambda` |
| Human transmission (`beta_j0_hum`, `a_*_j`, `b_*_j`, `p`, `alpha_*`) | `HumanToHuman` | `beta_jt_human`, `Lambda` |
| Environmental transmission (`beta_j0_env`, `theta_j`, `psi_jt`, `zeta_*`, `kappa`, `decay_*`) | `EnvToHuman`, `Environmental` | `beta_jt_env`, `delta_jt`, `Psi`, `W` |

If a parameter is at its inert (off) value the corresponding output
array still exists — it is just constant, zero, or otherwise
contribution-free. For example with `beta_j0_hum = 0` the
`HumanToHuman` component still runs, still allocates `Lambda`, and
still writes to `incidence_human` — but `Lambda` stays at zero and
`incidence_human` stays at zero, so the simulation just sees no
direct-transmission infections. A couple of off-switches are paired
rather than single-parameter: regime switching is disabled only when
`chi_endemic == chi_epidemic` *and* `mu_j_epidemic_factor == 0`; the
environmental forcing curve is canonically inert at `psi_jt = 1`
everywhere (not zeros). See the per-parameter
[reference pages](../reference/parameters/index.md) for the canonical
off-value of each knob.

## Stochasticity and reproducibility

All random draws use the same PRNG instance, constructed once in
`Model.__init__`:

```python
self.prng = seed_prng(
    parameters.seed if parameters.seed is not None else self.tinit.microsecond
)
```

`seed_prng` is `laser.core.random.seed`, which returns a
`numpy.random.Generator`. Every compartment component holds a reference
to `model` and pulls draws via `model.prng.binomial(...)`,
`model.prng.poisson(...)`, etc. There is no per-component RNG and no
hidden global state — the same `seed` always produces the same
trajectory on the same machine and same NumPy version.

This is what `misc/perf_baseline.py` exploits to maintain a bit-for-bit
baseline of the bundled SSA configuration: it runs `run_model(None)`
with a fixed seed and compares the resulting `model.people` /
`model.patches` arrays against a frozen reference. Any change to the
order of draws (re-ordering components, adding or removing a
`prng` call inside a tick, switching from Binomial to Bernoulli) will
break the baseline and surface as a CI failure on the perf job.

The corollary for callers: if you change the seed, you change every
draw in the run — there is no way to "vary just the births" without
varying everything else. For sensitivity sweeps you re-run the full
pipeline at each new seed.

## How it connects to the rest of the model

This page is the map. The deep dives are:

- [Transmission](transmission.md) — the math behind `Lambda`, `Psi`,
  and how `beta_j0_hum`, `beta_j0_env`, `theta_j`, `kappa`,
  `decay_days_*`, and `zeta_*` combine across `HumanToHuman`,
  `EnvToHuman`, and `Environmental`.
- [Seasonality](seasonality.md) — the two-mode harmonic that produces
  `beta_jt_human` and the suitability-driven derivation of `beta_jt_env`.
- [Mobility](mobility.md) — the gravity-model construction of `pi_ij`
  and the role of `tau_i`, `mobility_omega`, and `mobility_gamma`,
  including what falls out cleanly at single-patch.
- [Reporting and likelihood](reporting-and-likelihood.md) — the
  thinning from incidence to `reported_cases` / `reported_deaths`, the
  Negative-Binomial core of `calc_model_likelihood`, and the four
  shape-weighted terms that augment it.

If you arrived here from a Reference page wondering "where in the
pipeline does this parameter actually take effect?", the per-tick table
above is the answer — cross-walk the parameter to its consumer, then
read that component's source under `src/laser/cholera/metapop/`.

## See also

- [Parameter reference index](../reference/parameters/index.md) — every
  parameter, by group, with shape / dtype / off-value.
- [First-run tutorial](../tutorials/first-run.md) — runs the bundled
  defaults end-to-end and looks at the populated arrays.
- [Single-location tutorial](../tutorials/single-location.md) — builds
  the simplest possible parameter set from scratch and turns on one
  feature at a time.
- [Configurations / single-location](../configurations/single-location.md),
  [multi-admin](../configurations/multi-admin.md),
  [SSA baseline](../configurations/ssa-baseline.md) — the three
  reference configurations the project ships.
- The source of truth: `src/laser/cholera/metapop/model.py` (run loop
  and component wiring) and the per-component files
  (`susceptible.py`, `exposed.py`, `infectious.py`, `recovered.py`,
  `vaccinated.py`, `census.py`, `humantohuman.py`, `envtohuman.py`,
  `environmental.py`, `derivedvalues.py`, `analyzer.py`, `recorder.py`,
  `params.py`).
