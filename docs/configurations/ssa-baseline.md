# Sub-Saharan Africa, country level (bundled defaults)

The Sub-Saharan Africa baseline is the configuration shipped inside the package as `default_parameters.json`. It is what `metapop --seed 20240930` runs when you give it nothing else. The configuration treats each of forty Sub-Saharan African countries as a single patch, runs the model from 2023-02-01 to 2026-03-31, and exercises every dynamic component of the model — country-to-country mobility, two-mode seasonality, both transmission channels, regime switching, and reporting noise. Use it as the canonical reference point: every other configuration in this directory is derived by *turning features off* relative to this baseline.

## What's in this configuration

The baseline turns on every modelled mechanism. The list below records what is dynamically active, and — for the small number of mechanisms that are inert by default — which inert pattern from §4 of `misc/standalone-docs-plan.md` is in force.

ON:

- **Human-to-human transmission** — `beta_j0_hum` is a length-40 per-country vector of non-zero contact rates. `alpha_1` and `alpha_2` set the density-dependence on infectious and recovered compartments.
- **Environmental transmission** — `beta_j0_env` is a length-40 per-country vector of non-zero env-to-human transmission rates. `theta_j` carries per-country WASH attenuation, `psi_jt` carries the (T, 40) environmental suitability surface, and `kappa`, `decay_days_*`, `decay_shape_*`, `zeta_1`, `zeta_2`, `zeta_ratio` close out the environmental reservoir dynamics.
- **Mobility** — `tau_i` is a length-40 vector of non-zero per-country outflow fractions; `mobility_omega` and `mobility_gamma` are the gravity-model exponents that govern `pi_ij`. Together with `latitude` and `longitude` they couple the forty patches.
- **Seasonality** — the four amplitudes (`a_1_j`, `a_2_j`, `b_1_j`, `b_2_j`) carry non-zero per-country values; `p = 365` is the harmonic period. The seasonal multiplier varies across the year.
- **Vital dynamics** — `b_jt` and `d_jt` are non-zero (T, 40) matrices of per-country, per-tick birth and death rates derived from UN WPP. `mu_j_baseline`, `mu_j_slope`, and `mu_j_epidemic_factor` populate the regime-dependent CFR.
- **Regime switching** — `chi_endemic` and `chi_epidemic` differ, `mu_j_epidemic_factor` is non-zero, and `epidemic_threshold` is set; patches flip between endemic and epidemic dynamics based on observed reported incidence.
- **Reporting noise and lag** — `rho` and `rho_deaths` are below 1.0 (under-reporting), and `delta_reporting_cases` / `delta_reporting_deaths` introduce a tick lag between estimated and observed series.
- **Disease progression** — `iota`, `gamma_1`, `gamma_2`, `epsilon`, `sigma` are all at their bundled non-trivial values.

OFF (or omitted) in the bundled defaults:

- **Vaccination** — `nu_1_jt` and `nu_2_jt` are (T, 40) zero matrices and `V1_j_initial = V2_j_initial = 0` for every country. `phi_1`, `phi_2`, `omega_1`, `omega_2` are populated but inert because no doses are delivered. The hooks are there; the campaign is empty.
- **Likelihood scoring** — `calc_likelihood` is not set in the bundled JSON. The analyzer guards likelihood computation on `"calc_likelihood" in model.params and model.params.calc_likelihood`, so the baseline run does not score against observed data even though `reported_cases` and `reported_deaths` are present. Set `calc_likelihood = true` (and pick the four `weight_*` values) via `--over` to enable scoring.

## Files

- `src/laser/cholera/metapop/data/default_parameters.json` — the bundled configuration; the source of truth for this page.
- `src/laser/cholera/metapop/data/default_parameters.json.gz` — gzipped copy of the same JSON, distributed alongside the uncompressed form for size-sensitive contexts.
- `docs/configurations/code/ssa-baseline.json` — a thin pointer back to the bundled defaults so the wave-2 smoke test treats this configuration the same way as the other two.

This configuration does not need a `laser-init` extract: the country-level admin layer, populations, coordinates, and per-tick demographic rates were assembled upstream by the R [`MOSAIC`](https://github.com/InstituteforDiseaseModeling/MOSAIC-pkg) package from UN WPP, WorldPop, and WHO weekly cholera reports, then frozen into the bundled JSON. The `laser-init` workflow is the route for *new* country-and-admin-level extracts (see the [multi-admin configuration](multi-admin.md)).

## Running it

From the repository root, the canonical CLI invocation is:

```bash
metapop --seed 20240930
```

No `--params` flag is required because `get_parameters` falls back to the bundled `default_parameters.json` when no `paramsource` is supplied. Equivalently, from Python:

```python
from laser.cholera.metapop.model import run_model

run_model(None, seed=20240930)
```

To run the same configuration but pointing explicitly at the JSON on disk (useful when you want to diff your `--over` overrides against the baseline):

```bash
metapop --params src/laser/cholera/metapop/data/default_parameters.json --seed 20240930
```

```python
from pathlib import Path

from laser.cholera.metapop.model import run_model

params_path = Path("src/laser/cholera/metapop/data/default_parameters.json")
run_model(params_path, seed=20240930)
```

## What to expect

The run produces a (T, 40) panel of estimated and observed cases and deaths over roughly 1,155 ticks (2023-02-01 through 2026-03-31). Each country traces an endemic baseline modulated by the two-mode seasonal multiplier, with mobility carrying force-of-infection across country borders and the environmental reservoir delaying the response to seasonal forcing. A subset of countries — those with `mu_j_epidemic_factor` substantial enough and reported incidence above `epidemic_threshold` — flips into the epidemic regime during the high-transmission window, picks up the elevated CFR, and reverts when reported incidence falls back. Observed cases and deaths track the underlying estimated series scaled by `rho` / `rho_deaths` and lagged by `delta_reporting_*`. Because the bundled defaults omit `calc_likelihood`, the run finishes without scoring against `reported_cases` / `reported_deaths`; the observed series are written to the recorder for inspection but no log-likelihood is computed.

## Where this fits

This configuration is the anchor: the other two derive from it by progressive removal.

- [Single-location toy](single-location.md) collapses to one patch and zeros every optional mechanism — see what the model does when only human-to-human transmission contributes.
- [Multi-admin configuration](multi-admin.md) keeps the dynamic richness but swaps the country layer for sub-national admin units inside a single country, via a `laser-init` extract.

For parameter-level detail on what each field controls, see the Reference group pages:

- [Run identity and calendar](../reference/parameters/run-identity.md)
- [Initial compartment populations](../reference/parameters/initial-populations.md)
- [Vital dynamics](../reference/parameters/vital-dynamics.md)
- [Vaccination](../reference/parameters/vaccination.md)
- [Disease progression](../reference/parameters/disease-progression.md)
- [Reporting](../reference/parameters/reporting.md)
- [Regime switching](../reference/parameters/regime-switching.md)
- [Geography and mobility](../reference/parameters/geography-and-mobility.md)
- [Human-to-human transmission](../reference/parameters/human-transmission.md)
- [Environmental transmission](../reference/parameters/environmental-transmission.md)

For the model-level "why" the bundled defaults are shaped the way they are, see the Explanation pages (wave-5 stubs at present):

- [Model overview](../explanation/model-overview.md)
- [Transmission](../explanation/transmission.md)
- [Seasonality](../explanation/seasonality.md)
- [Mobility](../explanation/mobility.md)
- [Reporting and likelihood](../explanation/reporting-and-likelihood.md)

For task-oriented recipes that modify the baseline in place — turn on vaccination, switch on likelihood scoring, override a single field via `--over` — see the wave-4 how-to guides, starting with [Override parameters](../how-to/override-parameters.md).
