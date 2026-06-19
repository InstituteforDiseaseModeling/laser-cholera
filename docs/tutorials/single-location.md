# Build a single-location configuration from scratch

If you have already run the bundled defaults via the [first-run tutorial](first-run.md), the next thing to understand is the *shape* of a `laser-cholera` configuration: what each parameter does, what its inert ("off") value looks like, and how to flip one feature on at a time without rebaselining the whole file. This tutorial walks you through building the same configuration that ships at `docs/configurations/code/single-location.json` — but this time you build it yourself, parameter group by parameter group, via `get_parameters(mods=...)`. By the end you will have constructed a minimal SEIRV configuration with only human-to-human transmission active, run the model, and then turned seasonality, vaccination, and environmental transmission on one feature at a time so you can see each one move the dynamics.

## Step 1 — Start from the bundled defaults

`get_parameters(None)` loads the SSA-baseline parameter set bundled with the package. That set targets ~40 sub-Saharan countries over a multi-year window — far more than we want for a toy, but it is the right place to start because every required field is already populated and every shape constraint is already consistent.

```pycon
>>> from laser.cholera.metapop import get_parameters
>>> params = get_parameters(None)
>>> len(params.location_name)
40
>>> params.nticks
1155

```

We will progressively override fields with `mods` until what we have is a single-patch one-year toy.

## Step 2 — Override to a single patch

The smallest legal configuration has `len(location_name) == 1` and a one-year calendar. We override the calendar, the location, and every initial-compartment vector that has to match `npatches=1`.

```python
mods = {
    "location_name": ["TOY"],
    "date_start": "2024-01-01",
    "date_stop": "2024-12-31",
    "seed": 20240930,
    "N_j_initial": [100000],
    "S_j_initial": [99990],
    "E_j_initial": [0],
    "I_j_initial": [10],
    "R_j_initial": [0],
    "V1_j_initial": [0],
    "V2_j_initial": [0],
    "prop_S_initial": [0.9999],
    "prop_E_initial": [0.0],
    "prop_I_initial": [0.0001],
    "prop_R_initial": [0.0],
    "prop_V1_initial": [0.0],
    "prop_V2_initial": [0.0],
    "longitude": [0.0],
    "latitude": [0.0],
}
```

Note that, at this point, calling `get_parameters(mods=mods)` would fail validation: every `(nticks, npatches)` matrix in the bundled defaults still has `npatches = 40` columns, which no longer matches `len(location_name) == 1`. We fix that in Step 3 by setting each matrix to its inert form — the same step that turns every optional feature off.

## Step 3 — Turn off everything except human-to-human transmission

The point of this step is to land at a configuration where exactly one transmission channel is active and every other feature is at the canonical inert value from §4 of `misc/standalone-docs-plan.md`. We extend the `mods` dict; each group below is annotated with the one-line reason.

**Mobility.** A single patch cannot exchange exposure with itself, but the validator still demands `tau_i`, `mobility_omega`, and `mobility_gamma`. Zero `tau_i`; leave the two scalars at any positive value.

```python
import numpy as np
nticks = 366
npatches = 1
mods["tau_i"] = [0.0]
mods["mobility_omega"] = 1.0
mods["mobility_gamma"] = 1.0
```

**Environmental transmission.** Zeroing `beta_j0_env` cancels the env-to-human term. `psi_jt` is subtle: even when `beta_j0_env = 0`, the env-to-human component initializes a `(psi - psi_bar) / psi_bar` ratio (see `envtohuman.py`) — and `psi_bar` is the per-patch mean of `psi_jt`. A zeros matrix would divide by zero. The canonical inert value is therefore **ones, not zeros**.

```python
mods["beta_j0_env"] = [0.0]
mods["theta_j"] = [0.0]
mods["zeta_1"] = 0.0
mods["zeta_2"] = 0.0
mods["psi_jt"] = np.ones((nticks, npatches), dtype=np.float32)
mods["psi_star_a"] = [1.0]
mods["psi_star_b"] = [0.0]
mods["psi_star_z"] = [1.0]
mods["psi_star_k"] = [0.0]
```

**Seasonality.** Both `a_1_j`/`a_2_j` and `b_1_j`/`b_2_j` are sine amplitudes (not periods, despite the historical naming) — `b_1_j` multiplies `sin(2*pi*t/p)`, `b_2_j` multiplies `sin(4*pi*t/p)`. Zero all four amplitudes; keep `p` positive so the harmonic is well-defined.

```python
mods["a_1_j"] = [0.0]
mods["a_2_j"] = [0.0]
mods["b_1_j"] = [0.0]
mods["b_2_j"] = [0.0]
mods["p"] = 365
```

**Vaccination.** No doses delivered, no vaccine-immunity waning, no initial vaccinated population. `phi_1`, `phi_2`, and `nu_jt_sources` remain at the bundled defaults but never get to act because no doses flow.

```python
mods["nu_1_jt"] = np.zeros((nticks, npatches), dtype=np.float32)
mods["nu_2_jt"] = np.zeros((nticks, npatches), dtype=np.float32)
mods["omega_1"] = 0.0
mods["omega_2"] = 0.0
mods["V1_j_initial"] = [0]
mods["V2_j_initial"] = [0]
```

**Vital dynamics.** Closed population: no births, no non-disease deaths, no per-patch baseline mortality, and no epidemic mortality inflation. (`mu_j_epidemic_factor` also partners with regime switching below.)

```python
mods["b_jt"] = np.zeros((nticks, npatches), dtype=np.float32)
mods["d_jt"] = np.zeros((nticks, npatches), dtype=np.float32)
mods["mu_jt"] = np.zeros((nticks, npatches), dtype=np.float32)
mods["mu_j_baseline"] = [0.0]
mods["mu_j_slope"] = [0.0]
mods["mu_j_epidemic_factor"] = [0.0]
```

**WASH protection.** Already zeroed above for environmental transmission; recorded here for completeness.

```python
mods["theta_j"] = [0.0]
```

**Regime switching.** Zeroing `epidemic_threshold` alone is *not* enough to disable the feature — any positive `Ireported` would flip every patch into the epidemic branch. The canonical disable pattern is `chi_endemic == chi_epidemic` together with `mu_j_epidemic_factor == 0.0` (already set). With those two partners in place, `epidemic_threshold` has no effect on dynamics.

```python
mods["chi_endemic"] = 1.0
mods["chi_epidemic"] = 1.0
mods["epidemic_threshold"] = 0.0
```

**Reporting.** Observed equals estimated, no tick lag.

```python
mods["rho"] = 1.0
mods["rho_deaths"] = 1.0
mods["delta_reporting_cases"] = 0
mods["delta_reporting_deaths"] = 0
```

**Likelihood.** Scoring is meaningfully undefined without observed data, so we disable it and pass `(1, 0)`-shape empty arrays for the observed series.

```python
mods["calc_likelihood"] = False
mods["reported_cases"] = [[]]
mods["reported_deaths"] = [[]]
```

Now build the parameter set and confirm it lands where we intended:

```python
params = get_parameters(mods=mods)
assert params.location_name == ["TOY"]
assert params.nticks == 366
assert int(params.S_j_initial[0]) == 99990
assert int(params.I_j_initial[0]) == 10
assert float(params.beta_j0_hum[0]) == 0.30000001192092896
assert float(params.beta_j0_env[0]) == 0.0
assert float(params.tau_i[0]) == 0.0
```

(`beta_j0_hum` is the only force-of-infection coefficient left non-zero — that is what makes this run actually do something. We did not touch it, so it carries the bundled default of `0.3`.)

## Step 4 — Run the model

We run with the same 13-component pipeline that `tests/test_docs_configurations.py` uses for the smoke test. That ordering is the canonical one for a single-location run.

```python
from laser.cholera.metapop import (
    Analyzer,
    Census,
    DerivedValues,
    Environmental,
    EnvToHuman,
    Exposed,
    HumanToHuman,
    Infectious,
    Parameters,
    Recorder,
    Recovered,
    Susceptible,
    Vaccinated,
)
from laser.cholera.metapop.model import Model

model = Model(params, name="docs-tutorial-single-location")
model.components = [
    Susceptible, Exposed, Recovered, Infectious, Vaccinated,
    Census, HumanToHuman, EnvToHuman, Environmental,
    DerivedValues, Analyzer, Recorder, Parameters,
]
model.run()

infectious = model.people.I_sym + model.people.I_asym
print("peak infectious:", int(infectious.sum(axis=1).max()))
print("final susceptible:", int(model.people.S[-1, 0]))
```

Qualitatively: a small SIR-style outbreak. Ten seed infections in a closed population of 100,000 with `beta_j0_hum = 0.3`, recovery `gamma_1 = 0.1` (symptomatic) and `gamma_2 = 0.5` (asymptomatic), and a symptomatic fraction `sigma = 0.25` grow into an epidemic that peaks somewhere mid-window and dies out as susceptibles deplete. Because the population is closed and natural-immunity waning is slow (`epsilon = 0.0003`), there is no second wave inside the year.

## Step 5 — Turn on one feature at a time

Each sub-section starts from the `mods` dict you have already built and changes only the entries that matter for the feature being turned on. Re-run the model after each change.

### 5a. Add seasonality

Give the first harmonic some amplitude. `a_1_j` multiplies the cosine term and `b_1_j` the sine term at frequency `2*pi/p`; leaving `a_2_j` and `b_2_j` at zero keeps the second harmonic off.

```python
mods["a_1_j"] = [0.3]
mods["b_1_j"] = [0.1]
params = get_parameters(mods=mods)
assert float(params.a_1_j[0]) == 0.30000001192092896
assert float(params.b_2_j[0]) == 0.0
```

Re-run the model. The epidemic now rides under a seasonal envelope: the same baseline transmission, modulated up and down with period `p = 365`. With the seed in early January the first half of the year is suppressed relative to the constant case; the outbreak shifts later and peaks at a different height.

### 5b. Add vaccination

We deliver 50 dose-1 doses per day starting at tick 30, attenuated by per-dose efficacy `phi_1 = 0.7`. Dose-2 stays off.

```python
nu_1 = np.zeros((nticks, npatches), dtype=np.float32)
nu_1[30:, 0] = 50.0
mods["nu_1_jt"] = nu_1
mods["phi_1"] = 0.7
params = get_parameters(mods=mods)
assert float(params.nu_1_jt[0, 0]) == 0.0
assert float(params.nu_1_jt[30, 0]) == 50.0
assert float(params.phi_1) == 0.699999988079071
```

Re-run. The susceptible pool now drains by two pathways instead of one — infection plus vaccination — so the epidemic peak is lower and arrives sooner relative to the seasonal-only run, and the final susceptible count at the end of the year is smaller.

### 5c. Add environmental transmission

Turn on the env-to-human pathway by giving `beta_j0_env` a positive coefficient. `psi_jt` stays at the all-ones matrix you set in Step 3, so the environmental forcing curve is constant.

```python
mods["beta_j0_env"] = [0.05]
params = get_parameters(mods=mods)
assert float(params.beta_j0_env[0]) == 0.05000000074505806
```

Re-run. There are now two transmission pathways: direct human-to-human contact and exposure via the environmental reservoir that builds up from infectious shedders. The peak rises and broadens compared to the vaccination-only run, because the second pathway adds a long-tailed contribution that does not decay as quickly as direct contacts do.

## Where to next

- [Tutorial: multi-location country](multi-location-country.md) — scale up from one patch to many admin units in one country, and turn on mobility.
- [Reference › Parameters](../reference/parameters/index.md) — the full catalogue for every parameter you just set, with shape, dtype, valid range, and off-value.
- [Configurations › Single-location](../configurations/single-location.md) — the same configuration this tutorial built, shipped as a JSON file.
