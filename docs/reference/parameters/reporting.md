# Reporting

These parameters govern how the model's internal transitions (new symptomatic infections and disease deaths) are turned into observable time series, and how those observations are aligned against external data. The first four — `rho`, `rho_deaths`, `delta_reporting_cases`, `delta_reporting_deaths` — are scalar knobs that thin and lag the simulated streams, producing the `reported_cases` / `reported_deaths` outputs the analyzer scores. The last two — `reported_cases` and `reported_deaths` — are the observed-data arrays themselves: per-patch, per-time-step targets that the likelihood scorer compares the simulation against.

Together they form the bridge between the underlying SEIR dynamics in `infectious.py` and the likelihood machinery in `analyzer.py`. With all four scalar knobs at their off-values (`rho = 1`, `rho_deaths = 1`, `delta_reporting_cases = 0`, `delta_reporting_deaths = 0`) the reported series become same-tick, fully-observed copies of the internal counts (subject only to the regime-switching scaling described in [Regime switching](regime-switching.md)).

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
|---|---|---|---|---|
| [`rho`](#rho) | `scalar` | `float32` | `[0, 1]` | `1.0` |
| [`rho_deaths`](#rho_deaths) | `scalar` | `float32` | `[0, 1]` | `1.0` |
| [`delta_reporting_cases`](#delta_reporting_cases) | `scalar` | `int32` | `>= 0` | `0` |
| [`delta_reporting_deaths`](#delta_reporting_deaths) | `scalar` | `int32` | `>= 0` | `0` |
| [`reported_cases`](#reported_cases) | `(npatches, n_obs_timesteps)` | `float32` | shape-only check | n/a (not a feature toggle) |
| [`reported_deaths`](#reported_deaths) | `(npatches, n_obs_timesteps)` | `float32` | shape-only check | n/a (not a feature toggle) |

### `rho`

- **What it controls**: Per-case reporting probability for symptomatic cases — fraction of new symptomatic infections that show up in the `reported_cases` series.
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `[0, 1]`
- **Off-value**: `1.0` — every new symptomatic case is counted (the `Binomial(n, 1) == n` identity), leaving the regime-switching scaling as the only remaining transform between internal `new_symptomatic` and the reported series.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`rho_deaths`](#rho_deaths), [`delta_reporting_cases`](#delta_reporting_cases), [`chi_endemic`](regime-switching.md#chi_endemic), [`chi_epidemic`](regime-switching.md#chi_epidemic), [`epidemic_threshold`](regime-switching.md#epidemic_threshold), [`reported_cases`](#reported_cases)
- **Notes**: `infectious.py:267` draws `Binomial(new_symptomatic, rho)` then divides by `chi_eff`; with `rho = 1` every symptomatic case is counted, so the only remaining transform is the `chi_endemic` / `chi_epidemic` scaling. To get strict "observed == estimated symptomatic" also requires `chi_endemic == chi_epidemic == 1`.

### `rho_deaths`

- **What it controls**: Per-death reporting probability — fraction of disease deaths that show up in the `reported_deaths` series.
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `[0, 1]`
- **Off-value**: `1.0` — every disease death is counted, so the reported-deaths series equals the simulated disease-death count (up to the `delta_reporting_deaths` lag).
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`rho`](#rho), [`delta_reporting_deaths`](#delta_reporting_deaths), [`reported_deaths`](#reported_deaths)

### `delta_reporting_cases`

- **What it controls**: Integer tick lag between when a new symptomatic case occurs and when it is added to `reported_cases`.
- **Shape**: `scalar`
- **Dtype**: `int32`
- **Range**: `>= 0`
- **Off-value**: `0` — cases are written into the reported series on the same tick they occur (no lag).
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`rho`](#rho), [`delta_reporting_deaths`](#delta_reporting_deaths), [`epidemic_threshold`](regime-switching.md#epidemic_threshold)
- **Notes**: Also used at `infectious.py:196` to compute the `epidemic_flag` from a lagged `Isym` snapshot, so this parameter also shifts the regime-switching probe — not just the reporting series.

### `delta_reporting_deaths`

- **What it controls**: Integer tick lag between when a disease death occurs and when it is added to `reported_deaths`.
- **Shape**: `scalar`
- **Dtype**: `int32`
- **Range**: `>= 0`
- **Off-value**: `0` — deaths are written into the reported series on the same tick they occur (no lag).
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`rho_deaths`](#rho_deaths), [`delta_reporting_cases`](#delta_reporting_cases)

### `reported_cases`

- **What it controls**: Observed per-patch-per-time-step case counts used as the target series when computing the model likelihood.
- **Shape**: `(npatches, n_obs_timesteps)`
- **Dtype**: `float32`
- **Range**: shape-only check (type assert: must be `list` or `ndarray`; `NaN` allowed for missing weeks).
- **Off-value**: n/a (not a feature toggle) — `reported_cases` is observed data, not a switch. To run the model without scoring against case data, set `calc_likelihood` to `False` instead; the analyzer only consumes `reported_cases` when `calc_likelihood` is on.
- **Consumer code**: [`src/laser/cholera/metapop/analyzer.py`](../../reference/index.md)
- **Related parameters**: [`reported_deaths`](#reported_deaths), `calc_likelihood`, [`rho`](#rho), [`delta_reporting_cases`](#delta_reporting_cases), `weight_cases`
- **Notes**: Ingested via `handle_nan`, which converts non-numeric cells to `np.nan` (hence `float32`, not `int`). The only invariant enforced is type (list-of-lists or `ndarray`) in `dict_to_propertysetex` — no shape or range check in `validate_parameters`. The analyzer slices by `reported_cases.shape[1]` columns, confirming the `(npatches, n_obs_timesteps)` layout.

### `reported_deaths`

- **What it controls**: Observed per-patch-per-time-step death counts used as the target series when computing the model likelihood.
- **Shape**: `(npatches, n_obs_timesteps)`
- **Dtype**: `float32`
- **Range**: shape-only check (type assert: must be `list` or `ndarray`; `NaN` allowed for missing weeks).
- **Off-value**: n/a (not a feature toggle) — `reported_deaths` is observed data, not a switch. To run the model without scoring against death data, set `calc_likelihood` to `False` instead; the analyzer only consumes `reported_deaths` when `calc_likelihood` is on.
- **Consumer code**: [`src/laser/cholera/metapop/analyzer.py`](../../reference/index.md)
- **Related parameters**: [`reported_cases`](#reported_cases), `calc_likelihood`, [`rho_deaths`](#rho_deaths), [`delta_reporting_deaths`](#delta_reporting_deaths), `weight_deaths`
- **Notes**: Same ingestion path as `reported_cases` (`handle_nan` -> `float32` to accommodate `NaN` missing cells). No shape or range invariant in `validate_parameters`; only the type assert in `dict_to_propertysetex`. Consumed only by the analyzer when `calc_likelihood` is `True`.
