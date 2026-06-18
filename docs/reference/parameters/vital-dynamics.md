# Vital dynamics

This group controls the demographic and disease-mortality processes layered on top of the SEIR core: how many new susceptibles are born into each patch per tick, how aggressively a constant non-disease (background) hazard culls every live compartment, and how cholera-specific case fatality is built up from a baseline, an optional linear time-trend, and an epidemic-regime multiplier. Together these parameters decide whether the simulated population is closed (zero births, zero non-disease deaths, zero case fatality) or open with realistic demographics, and they shape the long-term replenishment of the susceptible pool that drives multi-year endemicity.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
| --- | --- | --- | --- | --- |
| [`b_jt`](#b_jt) | `(nticks, npatches)` | `np.float32` | `>= 0` | `0` (entire array) |
| [`d_jt`](#d_jt) | `(nticks, npatches)` | `np.float32` | `>= 0` | `0` (entire array) |
| [`mu_j_baseline`](#mu_j_baseline) | `(npatches,)` | `np.float32` | `>= 0` | `0` (vector of zeros) |
| [`mu_j_slope`](#mu_j_slope) | `(npatches,)` | `np.float32` | unconstrained | `0` (vector of zeros) |
| [`mu_j_epidemic_factor`](#mu_j_epidemic_factor) | `(npatches,)` | `np.float32` | `>= 0` | `0` (vector of zeros) |
| [`mu_jt`](#mu_jt) | `(nticks, npatches)` | `np.float32` | unconstrained | n/a (diagnostic only) |

### `b_jt`

- **What it controls**: Per-patch, per-tick crude birth rate; the Poisson mean for new susceptibles each day is `N * b_jt[tick]`.
- **Shape**: `(nticks, npatches)`
- **Dtype**: `np.float32`
- **Range**: `>= 0` (validator-enforced via `assert np.all(params.b_jt >= 0.0)`; the assert message says "b_jt rate values must be positive" but the check admits zero).
- **Off-value**: `0` (entire `(nticks, npatches)` array) — with the per-tick rate at zero the Poisson mean for new births is zero, so no new susceptibles are ever drawn into the population.
- **Consumer code**: [`src/laser/cholera/metapop/susceptible.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`d_jt`](#d_jt), [`N_j_initial`](initial-populations.md#n_j_initial)
- **Notes**: Stored as `(nticks, npatches)` after a transpose in `dict_to_propertysetex` — if the JSON ships it as `(num_nodes, num_ticks)` it is auto-transposed. The validator's assert message says "must be positive" but the implementation is `>= 0`, so zero is legal (which is what enables the closed-population off-switch). Births at tick `t` are drawn as `Poisson(N * b_jt[tick])` in `susceptible.py`; with `b_jt = 0` the mean is `0` and zero births are drawn.

### `d_jt`

- **What it controls**: Per-patch, per-tick non-disease (background) mortality rate applied to every live compartment (S, E, Isym, Iasym, R, V1, V2).
- **Shape**: `(nticks, npatches)`
- **Dtype**: `np.float32`
- **Range**: `>= 0` (validator-enforced via `assert np.all(params.d_jt >= 0.0)`; the assert message says "d_jt rate values must be positive" but the check admits zero).
- **Off-value**: `0` (entire `(nticks, npatches)` array) — every compartment caches `1 - exp(-d_jt)` and draws `Binomial(X, p)` non-disease deaths; with the rate at zero the probability is zero and the population is closed against background mortality.
- **Consumer code**: [`src/laser/cholera/metapop/susceptible.py`](../../reference/index.md), [`src/laser/cholera/metapop/exposed.py`](../../reference/index.md), [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/recovered.py`](../../reference/index.md), [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`b_jt`](#b_jt), [`mu_j_baseline`](#mu_j_baseline), [`mu_j_slope`](#mu_j_slope), [`mu_j_epidemic_factor`](#mu_j_epidemic_factor)
- **Notes**: Auto-transposed from `(num_nodes, num_ticks)` to `(nticks, npatches)` if needed. Each compartment caches `model.patches.non_disease_death_prob_jt = 1 - exp(-d_jt)` once at `check()` time and indexes by tick at runtime; with `d_jt = 0` the cache is all zeros and `Binomial(X, 0) = 0`, so non-disease deaths vanish in every compartment. Note the asymmetry with disease mortality, which is the `mu_*` family and lives only in `infectious.py`.

### `mu_j_baseline`

- **What it controls**: Per-patch baseline disease (cholera) mortality rate applied to the symptomatic infectious compartment; the multiplicative anchor of `mu_jt = mu_j_baseline * (1 + mu_j_slope * t/nticks) * (1 + mu_j_epidemic_factor * epidemic_flag)`.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: `>= 0` (validator-enforced via `assert np.all(params.mu_j_baseline >= 0)`).
- **Off-value**: `0` (vector of zeros) — with the baseline at zero the product `mu_jt` is zero regardless of slope or epidemic factor, so symptomatic disease deaths are drawn from `Binomial(Isym, 0) = 0` and disappear entirely.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`mu_j_slope`](#mu_j_slope), [`mu_j_epidemic_factor`](#mu_j_epidemic_factor), [`epidemic_threshold`](regime-switching.md#epidemic_threshold), [`delta_reporting_cases`](reporting.md#delta_reporting_cases)
- **Notes**: The plan's §4 does not enumerate a disease-mortality off-switch; this off-value is inferred from the multiplicative form in `infectious.py`. With `mu_j_baseline = 0`, the product `mu_jt = 0`, so `disease_deaths = Binomial(Isym, 1 - exp(0)) = 0` regardless of slope or epidemic factor. Only applies to symptomatic infectious (Isym), not asymptomatic (Iasym).

### `mu_j_slope`

- **What it controls**: Per-patch linear time-trend on disease mortality; `mu_jt` is scaled by `(1 + mu_j_slope * tick/nticks)` so positive values ramp mortality up over the run and negative values ramp it down.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained — only the shape is checked; the validator's comment notes "no range constraints on mu_j_slope".
- **Off-value**: `0` (vector of zeros) — with slope zero the time factor collapses to `1` and the simulated `mu_jt` reduces to `mu_j_baseline * (1 + mu_j_epidemic_factor * epidemic_flag)`, removing any drift over the run.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`mu_j_baseline`](#mu_j_baseline), [`mu_j_epidemic_factor`](#mu_j_epidemic_factor)
- **Notes**: This is the only parameter in this group without a non-negativity guard — a slope of `-1` would drive `mu_jt` negative at the end of the run, which would break the `np.expm1(-mu_jt)` semantics used to convert a rate into a per-tick probability. With `slope = 0` the time factor collapses to `1` exactly and `mu_jt` reduces to `mu_j_baseline * (1 + mu_j_epidemic_factor * epidemic_flag)`.

### `mu_j_epidemic_factor`

- **What it controls**: Per-patch multiplicative boost on disease mortality when the patch is flagged as being in an epidemic regime (reported symptomatic infectious exceeds `epidemic_threshold * N`).
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: `>= 0` (validator-enforced via `assert np.all(params.mu_j_epidemic_factor >= 0)`).
- **Off-value**: `0` (vector of zeros) — with the factor at zero the epidemic bracket collapses to `1.0` regardless of the regime flag, so disease mortality stays at the endemic level even during a declared outbreak.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`mu_j_baseline`](#mu_j_baseline), [`mu_j_slope`](#mu_j_slope), [`epidemic_threshold`](regime-switching.md#epidemic_threshold), [`delta_reporting_cases`](reporting.md#delta_reporting_cases), [`chi_endemic`](regime-switching.md#chi_endemic), [`chi_epidemic`](regime-switching.md#chi_epidemic)
- **Notes**: With `factor = 0` the second multiplicative bracket collapses to `1.0` regardless of `epidemic_flag`, so the epidemic regime does not amplify disease mortality. The epidemic flag is still computed using [`epidemic_threshold`](regime-switching.md#epidemic_threshold) and the reported (lagged) `Isym` value, but its product with the factor is zero so the regime-switch has no effect on mortality.

### `mu_jt`

- **What it controls**: Disease-mortality time-series stored on the parameter set for diagnostic plotting only — the simulation loop computes its own per-tick `mu_jt` from baseline / slope / epidemic-factor and ignores this array.
- **Shape**: `(nticks, npatches)` as stored (never validated or transposed).
- **Dtype**: `np.float32`
- **Range**: unconstrained — the validator does not check this field at all.
- **Off-value**: n/a — this is a diagnostic display copy, not a feature toggle; zeroing it has no effect on the simulation.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`mu_j_baseline`](#mu_j_baseline), [`mu_j_slope`](#mu_j_slope), [`mu_j_epidemic_factor`](#mu_j_epidemic_factor)
- **Notes**: Surprising: `mu_jt` is the only parameter in this group that is loaded by `dict_to_propertysetex` but has zero presence in `validate_parameters` — no shape check, no range check, no auto-transpose. It is read at exactly one site, `Parameters.plot()`, which calls `imshow(self.model.params.mu_jt.T, ...)`. The simulation's effective disease-mortality field is recomputed inline in `infectious.py` from [`mu_j_baseline`](#mu_j_baseline), [`mu_j_slope`](#mu_j_slope), and [`mu_j_epidemic_factor`](#mu_j_epidemic_factor); the stored `mu_jt` array is purely a precomputed display copy and can drift from the actual simulated values.
