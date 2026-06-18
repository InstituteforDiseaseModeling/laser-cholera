# Environmental transmission

These parameters control the second of the model's two transmission pathways: the indirect, water-borne route where infectious individuals shed *Vibrio cholerae* into a per-patch environmental reservoir `W` and susceptibles re-encounter that reservoir through a saturating force of infection. Together they govern the baseline reservoir-to-human rate (`beta_j0_env`), the seasonal modulation of that rate and of the reservoir's decay (`psi_jt`, `decay_*`), the WASH attenuation that reduces both transmission and shedding (`theta_j`), the per-compartment shedding rates (`zeta_1`, `zeta_2`), and the saturation half-constant (`kappa`). A handful of additional `psi_star_*`, `zeta_ratio`, and `decay_days_spread` keys are MOSAIC-side metadata that travel with the parameter set but are not read by any laser-cholera component.

To turn the entire environmental pathway off, set `beta_j0_env = np.zeros(npatches)` (and, if you also want the reservoir to stop accumulating, `zeta_1 = zeta_2 = 0`). See [Explanation › Transmission](../../explanation/transmission.md) for how these parameters compose into the environmental force of infection.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
|---|---|---|---|---|
| [`beta_j0_env`](#beta_j0_env) | `(npatches, 1)` | `float32` | `>= 0` | `np.zeros(npatches)` |
| [`theta_j`](#theta_j) | `(npatches,)` | `float32` | `[0, 1]` | `np.zeros(npatches)` |
| [`psi_jt`](#psi_jt) | `(nticks, npatches)` | `float32` | shape-only | `psi_jt = 1.0` everywhere |
| [`psi_star_a`](#psi_star_a) | `(npatches,)` | `float32` | unconstrained | n/a (not consumed) |
| [`psi_star_b`](#psi_star_b) | `(npatches,)` | `float32` | unconstrained | n/a (not consumed) |
| [`psi_star_z`](#psi_star_z) | `(npatches,)` | `float32` | unconstrained | n/a (not consumed) |
| [`psi_star_k`](#psi_star_k) | `(npatches,)` | `float32` | unconstrained | n/a (not consumed) |
| [`zeta_1`](#zeta_1) | scalar | `float32` | `>= 0` | `0.0` |
| [`zeta_2`](#zeta_2) | scalar | `float32` | `>= 0` | `0.0` |
| [`zeta_ratio`](#zeta_ratio) | scalar | Python `float` | unconstrained | n/a (not consumed) |
| [`kappa`](#kappa) | scalar | `float32` | `>= 0` | n/a (tuning, not a toggle) |
| [`decay_days_short`](#decay_days_short) | scalar | `float32` | `> 0` and `<= decay_days_long` | n/a (tuning, not a toggle) |
| [`decay_days_long`](#decay_days_long) | scalar | `float32` | `>= decay_days_short` | n/a (tuning, not a toggle) |
| [`decay_days_spread`](#decay_days_spread) | scalar | Python `int` | unconstrained | n/a (not consumed) |
| [`decay_shape_1`](#decay_shape_1) | scalar | `float32` | unconstrained (scipy needs `> 0`) | n/a (tuning, not a toggle) |
| [`decay_shape_2`](#decay_shape_2) | scalar | `float32` | unconstrained (scipy needs `> 0`) | n/a (tuning, not a toggle) |

## Parameters

### `beta_j0_env`

- **What it controls**: Per-patch baseline environmental-reservoir-to-human transmission rate; multiplier on the `W / (kappa + W)` saturation term in the environmental force of infection.
- **Shape**: `(npatches, 1)` — coerced 1-D vector reshaped to `(-1, 1)` by `dict_to_propertysetex` for broadcast.
- **Dtype**: `float32`
- **Range**: `>= 0` (validator: `"beta_j0_env values must be >= 0"`); the validator also checks `len(beta_j0_env) == npatches`.
- **Off-value**: `np.zeros(npatches)` — `beta_jt_env` is then identically zero, so no S to E environmental infections occur and the environmental pathway is fully off.
- **Consumer code**: [`src/laser/cholera/metapop/envtohuman.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`psi_jt`](#psi_jt), [`theta_j`](#theta_j), [`kappa`](#kappa), [`beta_j0_hum`](human-transmission.md#beta_j0_hum)
- **Notes**: Unusual shape — `dict_to_propertysetex` reshapes the 1-D `as_ndarray` result to `(npatches, 1)`, and the consumer uses `beta_j0_env.T` so that broadcasting against the `(nticks, npatches)` psi-modulation works out. With `beta_j0_env = 0` everywhere, `beta_jt_env` is identically zero, `Psi = 0`, and no S to E environmental infections occur.

### `theta_j`

- **What it controls**: Per-patch WASH (water, sanitation, and hygiene) coverage fraction; `(1 - theta_j)` attenuates both environmental transmission to humans and infectious shedding into the reservoir.
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: `[0, 1]` per element (validator: `"theta_j values must be in the range [0, 1]"`).
- **Off-value**: `np.zeros(npatches)` — disables WASH protection, so the `(1 - theta_j)` multiplier becomes 1 and environmental transmission and shedding run at their unattenuated rates. Note this *maximises* the environmental pathway rather than disabling it; to actually disable environmental transmission set [`beta_j0_env`](#beta_j0_env) to zero (and optionally [`zeta_1`](#zeta_1) and [`zeta_2`](#zeta_2) to zero as well).
- **Consumer code**: [`src/laser/cholera/metapop/envtohuman.py`](../index.md), [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`beta_j0_env`](#beta_j0_env), [`zeta_1`](#zeta_1), [`zeta_2`](#zeta_2)

### `psi_jt`

- **What it controls**: Per-tick, per-patch environmental suitability time series; modulates seasonal environmental transmission around its temporal mean and drives the suitability-to-decay map for the reservoir.
- **Shape**: `(nticks, npatches)` — auto-transposed from `(npatches, nticks)` input if needed.
- **Dtype**: `float32`
- **Range**: Shape-only check (validator: `"Shape of psi_jt ... does not match (nticks, npatches)"`); no numeric range is enforced (a `TODO - TBD` comment sits in `validate_parameters`).
- **Off-value**: `psi_jt = 1.0` everywhere — `beta_jt_env = beta_j0_env.T * (1 + (psi - psi_bar) / psi_bar)` collapses to `beta_j0_env` and `delta_jt` becomes time-invariant per patch. Any positive constant achieves the same effect since `psi - psi_bar = 0`, but `psi_jt` must not be zero (`psi_bar` is in the denominator); constant `1` is the safe canonical choice.
- **Consumer code**: [`src/laser/cholera/metapop/envtohuman.py`](../index.md), [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`beta_j0_env`](#beta_j0_env), [`decay_days_short`](#decay_days_short), [`decay_days_long`](#decay_days_long), [`decay_shape_1`](#decay_shape_1), [`decay_shape_2`](#decay_shape_2)

### `psi_star_a`

- **What it controls**: Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code.
- **Shape**: `(npatches,)` effectively — coerced via `np.array` but no validator shape check.
- **Dtype**: `float32`
- **Range**: unconstrained (validator does not check it).
- **Off-value**: n/a — not a feature toggle; the value is irrelevant in laser-cholera as long as [`psi_jt`](#psi_jt) is provided directly.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../index.md), [`src/laser/cholera/metapop/utils.py`](../index.md)
- **Related parameters**: [`psi_jt`](#psi_jt), [`psi_star_b`](#psi_star_b), [`psi_star_z`](#psi_star_z), [`psi_star_k`](#psi_star_k)
- **Notes**: Dead parameter w.r.t. dynamics: `dict_to_propertysetex` coerces it to a `float32` ndarray and `override_helper` accepts it as a CLI-unsupported key (vectors are not CLI-overridable), but nothing reads it during simulation. Likely a MOSAIC-side artefact retained for round-tripping.

### `psi_star_b`

- **What it controls**: Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code.
- **Shape**: `(npatches,)` effectively — coerced via `np.array` but no validator shape check.
- **Dtype**: `float32`
- **Range**: unconstrained (validator does not check it).
- **Off-value**: n/a — not a feature toggle; inert within laser-cholera.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../index.md), [`src/laser/cholera/metapop/utils.py`](../index.md)
- **Related parameters**: [`psi_jt`](#psi_jt), [`psi_star_a`](#psi_star_a), [`psi_star_z`](#psi_star_z), [`psi_star_k`](#psi_star_k)
- **Notes**: Same status as [`psi_star_a`](#psi_star_a) — coerced but never read by any simulation component.

### `psi_star_z`

- **What it controls**: Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code.
- **Shape**: `(npatches,)` effectively — coerced via `np.array` but no validator shape check.
- **Dtype**: `float32`
- **Range**: unconstrained (validator does not check it).
- **Off-value**: n/a — not a feature toggle; inert within laser-cholera.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../index.md), [`src/laser/cholera/metapop/utils.py`](../index.md)
- **Related parameters**: [`psi_jt`](#psi_jt), [`psi_star_a`](#psi_star_a), [`psi_star_b`](#psi_star_b), [`psi_star_k`](#psi_star_k)
- **Notes**: Same status as [`psi_star_a`](#psi_star_a) — coerced but never read.

### `psi_star_k`

- **What it controls**: Per-patch upstream-MOSAIC parameter for the suitability curve; carried in the parameter set but NOT consumed by any laser-cholera simulation code.
- **Shape**: `(npatches,)` effectively — coerced via `np.array` but no validator shape check.
- **Dtype**: `float32`
- **Range**: unconstrained (validator does not check it).
- **Off-value**: n/a — not a feature toggle; inert within laser-cholera.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../index.md), [`src/laser/cholera/metapop/utils.py`](../index.md)
- **Related parameters**: [`psi_jt`](#psi_jt), [`psi_star_a`](#psi_star_a), [`psi_star_b`](#psi_star_b), [`psi_star_z`](#psi_star_z)
- **Notes**: Same status as [`psi_star_a`](#psi_star_a) — coerced but never read.

### `zeta_1`

- **What it controls**: Per-infectious-symptomatic-person shedding rate into the environmental reservoir `W` per tick (Poisson rate).
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: `>= 0` (validator: `"zeta_1 value must be >= 0"`).
- **Off-value**: `0.0` — symptomatic individuals stop contributing to the reservoir; combined with [`zeta_2`](#zeta_2) `= 0` the entire shedding source disappears.
- **Consumer code**: [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`zeta_2`](#zeta_2), [`zeta_ratio`](#zeta_ratio), [`theta_j`](#theta_j)
- **Notes**: `environmental.py` shedding line: `shedding_sym = Poisson(zeta_1 * Isym)`. The `zeta_*` family is not listed as a stand-alone toggle in plan §4 — disabling environmental transmission via [`beta_j0_env`](#beta_j0_env) `= 0` already breaks the reservoir-to-human link; zeroing both `zeta_*` additionally breaks the human-to-reservoir link.

### `zeta_2`

- **What it controls**: Per-infectious-asymptomatic-person shedding rate into the environmental reservoir `W` per tick (Poisson rate).
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: `>= 0` (validator: `"zeta_2 value must be >= 0"`).
- **Off-value**: `0.0` — asymptomatic individuals stop contributing to the reservoir.
- **Consumer code**: [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`zeta_1`](#zeta_1), [`zeta_ratio`](#zeta_ratio), [`theta_j`](#theta_j)
- **Notes**: `environmental.py` shedding line: `shedding_asym = Poisson(zeta_2 * Iasym)`. Same logic as [`zeta_1`](#zeta_1).

### `zeta_ratio`

- **What it controls**: MOSAIC-side ratio used to derive `zeta_1`/`zeta_2` upstream; carried into the parameter set but NOT consumed by any laser-cholera simulation code.
- **Shape**: scalar (passed through unchanged; no coercion entry in `dict_to_propertysetex`).
- **Dtype**: Python `float` as loaded from JSON; never cast to `float32`.
- **Range**: unconstrained (no validator entry; no coercion entry).
- **Off-value**: n/a — not a feature toggle.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../index.md)
- **Related parameters**: [`zeta_1`](#zeta_1), [`zeta_2`](#zeta_2)
- **Notes**: Truly inert: not in the scalars/arrays coercion lists in `params.py`, not asserted by `validate_parameters`, and not read by any component. Listed only in `override_helper`'s mapping so CLI overrides will not reject it. Pure metadata.

### `kappa`

- **What it controls**: Half-saturation constant in the environmental force of infection `W / (kappa + W)`; larger `kappa` means more reservoir is needed before transmission saturates.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: `>= 0` (validator: `"kappa value must be >= 0"`).
- **Off-value**: n/a — this is a tuning constant, not a toggle. To disable environmental transmission set [`beta_j0_env`](#beta_j0_env) to zero rather than touching `kappa`.
- **Consumer code**: [`src/laser/cholera/metapop/envtohuman.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`beta_j0_env`](#beta_j0_env), [`theta_j`](#theta_j), [`psi_jt`](#psi_jt)
- **Notes**: Caveat — `kappa = 0` does not raise (the validator allows `>= 0`) but changes semantics: `W / (0 + W) = 1` wherever `W > 0` and `0 / 0` (NaN) at `W = 0`. Use [`beta_j0_env`](#beta_j0_env) `= 0` to disable the pathway; `kappa` becomes irrelevant once `beta_jt_env = 0` zeroes the whole numerator.

### `decay_days_short`

- **What it controls**: Shortest reservoir half-life (in days) at the high end of the suitability-to-decay map; `1 / decay_days_short` is the maximum decay rate.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: `> 0` (validator: `"decay_days_short value must be > 0"`); additionally `decay_days_short <= decay_days_long`.
- **Off-value**: n/a — this is a tuning constant, not a toggle.
- **Consumer code**: [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`decay_days_long`](#decay_days_long), [`decay_shape_1`](#decay_shape_1), [`decay_shape_2`](#decay_shape_2), [`psi_jt`](#psi_jt)
- **Notes**: Tunes the `delta_jt` range. Setting `decay_days_short == decay_days_long` collapses the suitability-to-decay map to a constant — effectively "no psi modulation of decay". The validator forbids `decay_days_short == 0`.

### `decay_days_long`

- **What it controls**: Longest reservoir half-life (in days) at the low end of the suitability-to-decay map; `1 / decay_days_long` is the minimum decay rate.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: `>= decay_days_short` (validator: `"decay_days_short ... value must be <= decay_days_long"`); no explicit `> 0` check but inherits `> 0` transitively via `decay_days_short`.
- **Off-value**: n/a — this is a tuning constant, not a toggle.
- **Consumer code**: [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`decay_days_short`](#decay_days_short), [`decay_shape_1`](#decay_shape_1), [`decay_shape_2`](#decay_shape_2), [`psi_jt`](#psi_jt)
- **Notes**: Paired with [`decay_days_short`](#decay_days_short). To make decay psi-independent, set `decay_days_short == decay_days_long`.

### `decay_days_spread`

- **What it controls**: MOSAIC-side scratch parameter describing the spread between fast and slow decay (in days); carried in the parameter set but NOT consumed by any laser-cholera simulation code.
- **Shape**: scalar (passed through unchanged; no coercion entry).
- **Dtype**: Python `int` from JSON; never cast.
- **Range**: unconstrained (no validator entry; no coercion entry).
- **Off-value**: n/a — not a feature toggle.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../index.md)
- **Related parameters**: [`decay_days_short`](#decay_days_short), [`decay_days_long`](#decay_days_long)
- **Notes**: Truly inert in laser-cholera. Listed in `override_helper`'s mapping only so CLI overrides will not reject it. Decay times are controlled by [`decay_days_short`](#decay_days_short) and [`decay_days_long`](#decay_days_long); `decay_days_spread` is metadata.

### `decay_shape_1`

- **What it controls**: Alpha shape parameter of the Beta CDF used to map `psi_jt` in `[0, 1]` onto the `[decay_days_short, decay_days_long]` interval.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: unconstrained by `validate_parameters` (no assert); `scipy.stats.beta.cdf` requires `> 0` at use time.
- **Off-value**: n/a — this is a tuning constant, not a toggle.
- **Consumer code**: [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`decay_shape_2`](#decay_shape_2), [`decay_days_short`](#decay_days_short), [`decay_days_long`](#decay_days_long), [`psi_jt`](#psi_jt)
- **Notes**: Coerced to `float32` but `validate_parameters` does not range-check it; misuse surfaces inside `scipy.stats.beta.cdf` at run time rather than at validation time.

### `decay_shape_2`

- **What it controls**: Beta shape parameter of the Beta CDF used to map `psi_jt` in `[0, 1]` onto the `[decay_days_short, decay_days_long]` interval.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: unconstrained by `validate_parameters` (no assert); `scipy.stats.beta.cdf` requires `> 0` at use time.
- **Off-value**: n/a — this is a tuning constant, not a toggle.
- **Consumer code**: [`src/laser/cholera/metapop/environmental.py`](../index.md), [`src/laser/cholera/metapop/params.py`](../index.md)
- **Related parameters**: [`decay_shape_1`](#decay_shape_1), [`decay_days_short`](#decay_days_short), [`decay_days_long`](#decay_days_long), [`psi_jt`](#psi_jt)
- **Notes**: Same caveat as [`decay_shape_1`](#decay_shape_1) — coerced but not range-validated; misuse surfaces in scipy at run time.
