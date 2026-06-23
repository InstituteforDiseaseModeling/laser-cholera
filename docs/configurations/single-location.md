# Single-location toy

The single-location toy is the smallest configuration that `laser-cholera` will accept and run. It collapses the metapopulation to one synthetic patch ("TOY") of 100,000 people seeded with 10 infectious cases, runs for 366 ticks (2024-01-01 through 2024-12-31), and disables every optional feature except human-to-human transmission. Use it to verify that the install works, to get a feel for the parameter shape conventions, and as a starting point you can mutate in about 30 seconds.

## What's in this configuration

The configuration leaves only one transmission channel active and zeros (or pairs) every other feature into its inert form. See the corrected inert-values table in `misc/standalone-docs-plan.md` (§4) for the canonical off-values referenced below.

ON:

- **Human-to-human transmission** — `beta_j0_hum = [0.3]`, `alpha_1 = 0.27`, `alpha_2 = 0.5`. This is the only force-of-infection term that actually contributes.

OFF:

- **Mobility** — `tau_i = [0.0]` zeros the per-patch outflow; `mobility_omega = mobility_gamma = 1.0` are still required by `validate_parameters` but have no effect when `tau_i` is zero.
- **Environmental transmission** — `beta_j0_env = [0.0]` disables the env-to-human term. `zeta_1 = zeta_2 = 0.0`, `theta_j = [0.0]`, and `psi_jt` is a 366×1 *ones* matrix — the canonical inert value per §4 of the plan, since a zeros matrix would NaN-out the `(psi - psi_bar) / psi_bar` initialization in `envtohuman.py:66` even though `beta_j0_env = 0` would cancel the multiplier downstream. `kappa`, `decay_days_*`, and `zeta_ratio` carry bundled defaults but never feed dynamics.
- **Seasonality** — all four amplitudes (`a_1_j`, `a_2_j`, `b_1_j`, `b_2_j`) are `[0.0]` and `p = 365`. With every amplitude zero, the seasonal multiplier collapses to a constant 1.0; the period stays positive so the divide in the harmonic is well-defined.
- **Vaccination** — `nu_1_jt` and `nu_2_jt` are 366x1 zero matrices, `omega_1 = omega_2 = 0.0`, and `V1_j_initial = V2_j_initial = 0`. `phi_1`, `phi_2`, and `nu_jt_sources` are left at the bundled values but have no effect because no doses are delivered.
- **WASH protection** — `theta_j = [0.0]` removes any WASH attenuation on the (already-disabled) environmental term.
- **Vital dynamics** — `b_jt`, `d_jt`, and `mu_jt` are 366x1 zero matrices; `mu_j_baseline`, `mu_j_slope`, and `mu_j_epidemic_factor` are all `[0.0]`. The population is closed: no births, no non-disease deaths.
- **Regime switching** — `chi_endemic = chi_epidemic = 1.0` paired with `mu_j_epidemic_factor = [0.0]`. Both partners of the corrected disable pattern are set, so `epidemic_threshold = 0.0` has no dynamic consequence.
- **Reporting noise** — `rho = rho_deaths = 1.0` (observed = estimated) and `delta_reporting_cases = delta_reporting_deaths = 0` (no lag).
- **Likelihood scoring** — `calc_likelihood = false`; `reported_cases` and `reported_deaths` are `[[]]` and parse to a `(1, 0)` `float32` array via `handle_nan`. Scoring is meaningfully undefined here because there is no observed data to score against.

## Files

- `docs/configurations/code/single-location.json` — the configuration consumed by `metapop` and `get_parameters`.

No upstream extract is needed; the single patch is synthetic. (The multi-admin configuration is the one that ships a frozen `laser-init` extract under `docs/configurations/code/`.)

## Running it

From the repository root:

```bash
metapop --params docs/configurations/code/single-location.json --seed 20240930
```

Equivalently, from Python:

```python
from pathlib import Path

from laser.cholera.metapop.model import run_model

params_path = Path("docs/configurations/code/single-location.json")
run_model(params_path, seed=20240930)
```

For comparison, the SSA baseline (the bundled `default_parameters.json`) runs without a `--params` flag:

```bash
metapop --seed 20240930
```

## What to expect

With one patch of 100,000 people, 10 seed infections, `beta_j0_hum = 0.3`, recovery rate `gamma_1 = 0.1` (symptomatic) and `gamma_2 = 0.5` (asymptomatic), and a symptomatic fraction `sigma = 0.25`, the run produces a single self-contained outbreak. Cases grow exponentially from the seed, peak somewhere inside the 366-day window, and decay as susceptibles are depleted. Because the population is closed (`b_jt = d_jt = 0`) and natural-immunity waning is slow (`epsilon = 0.0003`), the trajectory does not re-bloom inside one year. No environmental reservoir builds up, no doses are delivered, no patches exchange exposure, and reported cases exactly equal estimated cases tick-for-tick.

## Where this fits

This is the *minimal* configuration. Every other configuration in `docs/configurations/` is derived by turning features on:

- To enable vaccination, set `nu_1_jt` and `nu_2_jt` to non-zero values; see [how-to/enable-vaccination.md](../how-to/enable-vaccination.md).
- To enable mobility, expand `location_name` to multiple patches and set `tau_i` to non-zero values; see [how-to/configure-mobility.md](../how-to/configure-mobility.md) and the [multi-admin configuration](multi-admin.md).
- To enable seasonality, give the amplitude parameters non-zero values; see [how-to/enable-seasonality.md](../how-to/enable-seasonality.md).
- To enable likelihood scoring, supply `reported_cases` and `reported_deaths` and set `calc_likelihood = true`; see [how-to/calibrate-and-score.md](../how-to/calibrate-and-score.md).

For parameter-level detail, see the Reference group pages:

- [Run identity & calendar](../reference/parameters/run-identity.md)
- [Initial compartment populations](../reference/parameters/initial-populations.md)
- [Vital dynamics](../reference/parameters/vital-dynamics.md)
- [Vaccination](../reference/parameters/vaccination.md)
- [Disease progression](../reference/parameters/disease-progression.md)
- [Reporting](../reference/parameters/reporting.md)
- [Regime switching](../reference/parameters/regime-switching.md)
- [Geography & mobility](../reference/parameters/geography-and-mobility.md)
- [Human-to-human transmission](../reference/parameters/human-transmission.md)
- [Environmental transmission](../reference/parameters/environmental-transmission.md)

For the model-level "why", see the Explanation pages (wave 5 stubs):

- [Model overview](../explanation/model-overview.md)
- [Transmission](../explanation/transmission.md)
- [Mobility](../explanation/mobility.md)

And for a hand-held walkthrough that builds this same configuration field-by-field via `get_parameters(mods=...)`, see the tutorial (wave 3 stub):

- [Tutorial: single-location](../tutorials/single-location.md)
