# Regime switching

The regime-switching parameters let the model behave differently depending on whether a patch is in an endemic or epidemic state. A per-patch symptomatic-infected fraction is compared to a threshold; patches above the threshold are treated as in an epidemic regime, which alters both the reporting-rate divisor (`chi_endemic` vs. `chi_epidemic`) and — via the partner parameter [`mu_j_epidemic_factor`](vital-dynamics.md#mu_j_epidemic_factor) — the disease-mortality multiplier. The optional `epidemic_peaks` table is not consumed by the simulation dynamics; it carries observed peak dates per location that are scored by the likelihood machinery.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
|---|---|---|---|---|
| [`chi_endemic`](#chi_endemic) | scalar | `np.float32` | unconstrained | `chi_endemic == chi_epidemic` (e.g. both `1.0`) |
| [`chi_epidemic`](#chi_epidemic) | scalar | `np.float32` | unconstrained | `chi_epidemic == chi_endemic` (e.g. both `1.0`) |
| [`epidemic_threshold`](#epidemic_threshold) | scalar or `(npatches,)` | `np.float32` | `>= 0` | set `chi_endemic == chi_epidemic` **and** `mu_j_epidemic_factor == 0` |
| [`epidemic_peaks`](#epidemic_peaks) | `(num_peaks, 3)` DataFrame | pandas (object / int) | columns `iso_code`, `peak_date` required if field present | omit the field entirely |

## Parameters

### `chi_endemic`

- **What it controls**: Reporting-rate divisor used when the local symptomatic-infected fraction is below `epidemic_threshold` (endemic regime); inflates estimated cases via division.
- **Shape**: `scalar`
- **Dtype**: `np.float32`
- **Range**: unconstrained (cast to `np.float32` in `dict_to_propertysetex`; no validator-enforced bound in `validate_parameters`).
- **Off-value**: `chi_endemic == chi_epidemic` (e.g. both `1.0`) — collapses the regime switch so the reporting branch is a no-op regardless of `epidemic_threshold`; an additional value of `1.0` also makes the reporting divisor itself a no-op.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`chi_epidemic`](#chi_epidemic), [`epidemic_threshold`](#epidemic_threshold), [`rho`](reporting.md#rho), [`delta_reporting_cases`](reporting.md#delta_reporting_cases)
- **Notes**: Consumed in `infectious.py` as `np.where(infected_fraction < epidemic_threshold, chi_endemic, chi_epidemic)`, then `reported_cases += round(binomial(new_symptomatic, rho) / chi_eff)`. The bundled default value `0.5` inflates reported cases by `1 / 0.5 = 2x` in the endemic regime.

### `chi_epidemic`

- **What it controls**: Reporting-rate divisor used when the local symptomatic-infected fraction is at or above `epidemic_threshold` (epidemic regime); inflates estimated cases via division.
- **Shape**: `scalar`
- **Dtype**: `np.float32`
- **Range**: unconstrained (cast to `np.float32` in `dict_to_propertysetex`; no validator-enforced bound in `validate_parameters`).
- **Off-value**: `chi_epidemic == chi_endemic` (e.g. both `1.0`) — collapses the regime switch so the reporting branch is a no-op regardless of `epidemic_threshold`; an additional value of `1.0` also makes the reporting divisor itself a no-op.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`chi_endemic`](#chi_endemic), [`epidemic_threshold`](#epidemic_threshold), [`rho`](reporting.md#rho), [`delta_reporting_cases`](reporting.md#delta_reporting_cases)
- **Notes**: Bundled default value `0.75` (higher reporting under epidemic awareness than the endemic default of `0.5`). Setting `chi_epidemic == chi_endemic` makes the regime branch in `infectious.py` a no-op regardless of `epidemic_threshold`.

### `epidemic_threshold`

- **What it controls**: Per-patch (or global) symptomatic-infected fraction above which the patch is treated as in an epidemic regime, switching both the disease-mortality multiplier and the reporting-rate divisor.
- **Shape**: `scalar` or `(npatches,)`
- **Dtype**: `np.float32` (scalar `np.float32` if scalar input; `np.ndarray` of `np.float32` if list / array input).
- **Range**: `>= 0` (scalar branch asserts `params.epidemic_threshold >= 0`; array branch asserts `np.all(params.epidemic_threshold >= 0)`; raises `RuntimeError` if the value is neither a scalar nor an `np.ndarray`).
- **Off-value**: **Verified off-value (corrects plan §4):** set `chi_endemic == chi_epidemic` **and** [`mu_j_epidemic_factor`](vital-dynamics.md#mu_j_epidemic_factor) `== 0` to fully collapse both regime effects; any `epidemic_threshold` value then has no effect. The plan's "`epidemic_threshold = 0` (or large)" is not sufficient on its own: with `epidemic_threshold = 0` and any positive `Ireported`, the epidemic flag is `True` everywhere and `mu_jt` picks up the full `mu_j_epidemic_factor` inflation; with a very large threshold the flag is `False` everywhere (always endemic) but `chi_endemic` still applies. The plan's parenthetical `chi_endemic = chi_epidemic = 1.0` is the necessary half of the fix, and note that `mu_j_epidemic_factor` lives in the vital-dynamics group, not in regime-switching.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`chi_endemic`](#chi_endemic), [`chi_epidemic`](#chi_epidemic), [`mu_j_epidemic_factor`](vital-dynamics.md#mu_j_epidemic_factor), [`delta_reporting_cases`](reporting.md#delta_reporting_cases)
- **Notes**: Two distinct uses with different threshold operands: (1) `infectious.py:198` compares `Ireported` (count) to `epidemic_threshold * N`, so the threshold is interpreted as a fraction of population for the mortality switch; (2) `infectious.py:265` compares `infected_fraction = Isym / N` (fraction) to `epidemic_threshold` directly for the reporting-rate switch. The two checks use the same threshold value but with different left-hand-side scalings (`Ireported` vs. `Ireported / N`) — functionally consistent because both reduce to comparing `Isym / N` against the threshold. The field is commented out of the scalars and arrays tables in `dict_to_propertysetex` (lines 466 and 499) and handled in a dedicated branch (lines 545–552).

### `epidemic_peaks`

- **What it controls**: Observed epidemic peak dates per ISO code used by the likelihood scorer to evaluate peak-timing and peak-magnitude shape terms; not consumed by the simulation dynamics.
- **Shape**: `(num_peaks, 3)` DataFrame with columns `iso_code`, `peak_date`, `loc_idx` (the `loc_idx` column is added in `dict_to_propertysetex` by looking up `iso_code` in `location_name`).
- **Dtype**: pandas DataFrame (`iso_code`: object / str; `peak_date`: object / str — not parsed to `datetime` here; `loc_idx`: int).
- **Range**: shape-only check — the validator only requires the field to be convertible to a DataFrame and to contain `iso_code` and `peak_date` columns; `dict_to_propertysetex` additionally asserts every `iso_code` is present in `location_name`.
- **Off-value**: `omit the field entirely` — the validator branch is gated on `if "epidemic_peaks" in params`, so dropping the key skips all checks; equivalently, setting `weight_peak_timing` `=` `weight_peak_magnitude` `= 0` skips the peak-shape branch inside `calc_model_likelihood`.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/analyzer.py`](../../reference/index.md), [`src/laser/cholera/calc_model_likelihood.py`](../../reference/index.md)
- **Related parameters**: `weight_peak_timing`, `weight_peak_magnitude`, `sigma_peak_time`, `sigma_peak_log`, [`reported_cases`](reporting.md#reported_cases), [`date_start`](run-identity.md#date_start), [`date_stop`](run-identity.md#date_stop)
- **Notes**: Optional field. Not consumed in the simulation dynamics at all — only flows through `analyzer.py` (line 74) into `calc_model_likelihood` via the analyzer's likelihood-computation path. `dict_to_propertysetex` augments the DataFrame with a `loc_idx` column (line 565) for downstream consumers, but `validate_parameters` only checks the `iso_code` / `peak_date` columns. The R upstream's in-window peak filter is applied inside `calc_model_likelihood` (post-port semantics, commit `f8e3b38`).
