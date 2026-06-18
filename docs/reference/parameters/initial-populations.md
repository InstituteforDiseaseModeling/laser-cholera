# Initial compartment populations

These parameters set the tick-0 state of every patch: how many people start in each SEIR compartment (`S_j_initial`, `E_j_initial`, `I_j_initial`, `R_j_initial`), how many start vaccinated (`V1_j_initial`, `V2_j_initial`), and a total-population reference (`N_j_initial`). Together they fully determine the initial-condition vector that the metapop components consume at `tick = 0` — including the symptomatic / asymptomatic split of the initial infectious pool (via [`sigma`](disease-progression.md#sigma)) and the vaccine-on / vaccine-off behaviour at the start of the run.

A parallel set of `prop_*_initial` arrays carries per-patch population fractions for the same compartments. These are upstream-MOSAIC artifacts: they are type-coerced into the parameter object but no live consumer in `src/laser/cholera/` reads them. The absolute-count `*_j_initial` arrays are what actually seed the model. They are documented here for inventory completeness and to flag that editing them has no effect on the simulation.

All seven absolute-count arrays are `(npatches,)` `uint32` vectors with the same `>= 0` validator assertion (except `N_j_initial`, which is not shape- or range-checked). All seven `prop_*_initial` arrays are `(npatches,)` `float32` vectors with no validator constraint.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
|---|---|---|---|---|
| [`N_j_initial`](#n_j_initial) | `(npatches,)` | `uint32` | unconstrained | n/a |
| [`S_j_initial`](#s_j_initial) | `(npatches,)` | `uint32` | `>= 0` | n/a |
| [`E_j_initial`](#e_j_initial) | `(npatches,)` | `uint32` | `>= 0` | n/a |
| [`I_j_initial`](#i_j_initial) | `(npatches,)` | `uint32` | `>= 0` | n/a |
| [`R_j_initial`](#r_j_initial) | `(npatches,)` | `uint32` | `>= 0` | n/a |
| [`V1_j_initial`](#v1_j_initial) | `(npatches,)` | `uint32` | `>= 0` | `0` |
| [`V2_j_initial`](#v2_j_initial) | `(npatches,)` | `uint32` | `>= 0` | `0` |
| [`prop_S_initial`](#prop_s_initial) | `(npatches,)` | `float32` | unconstrained | n/a |
| [`prop_E_initial`](#prop_e_initial) | `(npatches,)` | `float32` | unconstrained | n/a |
| [`prop_I_initial`](#prop_i_initial) | `(npatches,)` | `float32` | unconstrained | n/a |
| [`prop_R_initial`](#prop_r_initial) | `(npatches,)` | `float32` | unconstrained | n/a |
| [`prop_V1_initial`](#prop_v1_initial) | `(npatches,)` | `float32` | unconstrained | n/a |
| [`prop_V2_initial`](#prop_v2_initial) | `(npatches,)` | `float32` | unconstrained | n/a |

### `N_j_initial`

- **What it controls**: Total initial population per patch (intended as the gravity-model destination mass when reconstructing the mobility matrix; the live code now sums the per-compartment initials instead).
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`; cast to `uint32` implicitly enforces `>= 0`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/` — see Notes.
- **Related parameters**: [`S_j_initial`](#s_j_initial), [`E_j_initial`](#e_j_initial), [`I_j_initial`](#i_j_initial), [`R_j_initial`](#r_j_initial), [`V1_j_initial`](#v1_j_initial), [`V2_j_initial`](#v2_j_initial), [`location_name`](run-identity.md#location_name), [`nu_1_jt`](vaccination.md#nu_1_jt), [`nu_2_jt`](vaccination.md#nu_2_jt)
- **Notes**: Has no live consumer in `src/laser/cholera/` — the only references in `params.py` are commented-out vaccination caps (`nu_1_jt <= N_j_initial / 7`) and the override-helper unsupported-key list. The gravity-model code in `utils.py` builds `N` from the sum of S/E/I/R/V1/V2 initials, not from `N_j_initial`. The validator does not even shape-check it. Document its value if you want it to match the sum of the seeded compartments, but be aware editing it has no runtime effect.

### `S_j_initial`

- **What it controls**: Per-patch initial number of susceptible people seeded into the S compartment at tick 0.
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: `>= 0` (validator-enforced; `assert np.all(params.S_j_initial >= 0)`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: [`src/laser/cholera/metapop/susceptible.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/envtohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/environmental.py`](../../reference/index.md), [`src/laser/cholera/metapop/exposed.py`](../../reference/index.md), [`src/laser/cholera/metapop/recovered.py`](../../reference/index.md), [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/census.py`](../../reference/index.md), [`src/laser/cholera/metapop/analyzer.py`](../../reference/index.md)
- **Related parameters**: [`N_j_initial`](#n_j_initial), [`E_j_initial`](#e_j_initial), [`I_j_initial`](#i_j_initial), [`R_j_initial`](#r_j_initial), [`V1_j_initial`](#v1_j_initial), [`V2_j_initial`](#v2_j_initial), [`prop_S_initial`](#prop_s_initial)
- **Notes**: Also used as a ranking key (`np.argsort(params.S_j_initial)[-10:]`) by many components to pick the ten most populous patches for plotting — an initial-condition value persists into runtime diagnostics, so changing it will reshuffle which patches appear in the default plots.

### `E_j_initial`

- **What it controls**: Per-patch initial number of exposed (latent, not yet infectious) people seeded into the E compartment at tick 0.
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: `>= 0` (validator-enforced; `assert np.all(params.E_j_initial >= 0)`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: [`src/laser/cholera/metapop/exposed.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`S_j_initial`](#s_j_initial), [`I_j_initial`](#i_j_initial), [`iota`](disease-progression.md#iota), [`prop_E_initial`](#prop_e_initial)

### `I_j_initial`

- **What it controls**: Per-patch initial number of infectious people; split by [`sigma`](disease-progression.md#sigma) into symptomatic (`Isym`) and asymptomatic (`Iasym`) at tick 0.
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: `>= 0` (validator-enforced; `assert np.all(params.I_j_initial >= 0)`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`sigma`](disease-progression.md#sigma), [`S_j_initial`](#s_j_initial), [`E_j_initial`](#e_j_initial), [`gamma_1`](disease-progression.md#gamma_1), [`gamma_2`](disease-progression.md#gamma_2), [`prop_I_initial`](#prop_i_initial)
- **Notes**: `infectious.py` lines 83-84 seed the two infectious branches as `Isym[0] = round(sigma * I_j_initial)` and `Iasym[0] = I_j_initial - Isym[0]`. This confirms the `sigma` convention used elsewhere: `sigma = 1` -> all symptomatic, `sigma = 0` -> all asymptomatic.

### `R_j_initial`

- **What it controls**: Per-patch initial number of recovered (immune) people seeded into the R compartment at tick 0.
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: `>= 0` (validator-enforced; `assert np.all(params.R_j_initial >= 0)`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: [`src/laser/cholera/metapop/recovered.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`S_j_initial`](#s_j_initial), [`epsilon`](disease-progression.md#epsilon), [`prop_R_initial`](#prop_r_initial)

### `V1_j_initial`

- **What it controls**: Per-patch initial number of one-dose-vaccinated people seeded into the V1 compartment at tick 0.
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: `>= 0` (validator-enforced; `assert np.all(params.V1_j_initial >= 0)`).
- **Off-value**: `0` (vector of zeros) — combined with `nu_1_jt = 0` this is the vaccination-off recipe from the plan §4 inert-values table; `vaccinated.py` line 83 reads `model.people.V1[0] = model.params.V1_j_initial`, so a zero seed yields an empty V1 compartment at tick 0 with no inflow when the per-tick first-dose rate is also zero.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`V2_j_initial`](#v2_j_initial), [`nu_1_jt`](vaccination.md#nu_1_jt), [`phi_1`](vaccination.md#phi_1), [`omega_1`](vaccination.md#omega_1), [`prop_V1_initial`](#prop_v1_initial)

### `V2_j_initial`

- **What it controls**: Per-patch initial number of two-dose-vaccinated people seeded into the V2 compartment at tick 0.
- **Shape**: `(npatches,)`
- **Dtype**: `uint32`
- **Range**: `>= 0` (validator-enforced; `assert np.all(params.V2_j_initial >= 0)`).
- **Off-value**: `0` (vector of zeros) — same vaccination-off pairing as [`V1_j_initial`](#v1_j_initial); `vaccinated.py` line 84 reads `model.people.V2[0] = model.params.V2_j_initial`, so a zero seed combined with `nu_2_jt = 0` keeps the V2 compartment empty throughout the run.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`V1_j_initial`](#v1_j_initial), [`nu_2_jt`](vaccination.md#nu_2_jt), [`phi_2`](vaccination.md#phi_2), [`omega_2`](vaccination.md#omega_2), [`prop_V2_initial`](#prop_v2_initial)

### `prop_S_initial`

- **What it controls**: Per-patch initial fraction of population in the S compartment (informational / upstream-MOSAIC artifact; not read by the model itself).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/`.
- **Related parameters**: [`S_j_initial`](#s_j_initial), [`prop_E_initial`](#prop_e_initial), [`prop_I_initial`](#prop_i_initial), [`prop_R_initial`](#prop_r_initial), [`prop_V1_initial`](#prop_v1_initial), [`prop_V2_initial`](#prop_v2_initial)
- **Notes**: Declared in the `arrays` list in `dict_to_propertysetex` (so it is type-coerced when present) but has no live consumer — the only other occurrences are the override-helper unsupported-key list (`utils.py`) and `default_parameters.json`. The absolute-count [`S_j_initial`](#s_j_initial) is what `susceptible.py` seeds. Likely a vestigial MOSAIC field; included here for inventory completeness.

### `prop_E_initial`

- **What it controls**: Per-patch initial fraction of population in the E compartment (informational / upstream-MOSAIC artifact; not read by the model).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/`.
- **Related parameters**: [`E_j_initial`](#e_j_initial), [`prop_S_initial`](#prop_s_initial), [`prop_I_initial`](#prop_i_initial), [`prop_R_initial`](#prop_r_initial), [`prop_V1_initial`](#prop_v1_initial), [`prop_V2_initial`](#prop_v2_initial)
- **Notes**: No live consumer; converted to `float32` by `dict_to_propertysetex` but never read. The absolute count [`E_j_initial`](#e_j_initial) is what `exposed.py` seeds.

### `prop_I_initial`

- **What it controls**: Per-patch initial fraction of population in the I compartment (informational / upstream-MOSAIC artifact; not read by the model).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/`.
- **Related parameters**: [`I_j_initial`](#i_j_initial), [`sigma`](disease-progression.md#sigma), [`prop_S_initial`](#prop_s_initial), [`prop_E_initial`](#prop_e_initial), [`prop_R_initial`](#prop_r_initial), [`prop_V1_initial`](#prop_v1_initial), [`prop_V2_initial`](#prop_v2_initial)
- **Notes**: No live consumer; converted to `float32` by `dict_to_propertysetex` but never read. `infectious.py` seeds from the absolute count [`I_j_initial`](#i_j_initial).

### `prop_R_initial`

- **What it controls**: Per-patch initial fraction of population in the R compartment (informational / upstream-MOSAIC artifact; not read by the model).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/`.
- **Related parameters**: [`R_j_initial`](#r_j_initial), [`prop_S_initial`](#prop_s_initial), [`prop_E_initial`](#prop_e_initial), [`prop_I_initial`](#prop_i_initial), [`prop_V1_initial`](#prop_v1_initial), [`prop_V2_initial`](#prop_v2_initial)
- **Notes**: No live consumer; converted to `float32` by `dict_to_propertysetex` but never read.

### `prop_V1_initial`

- **What it controls**: Per-patch initial fraction of population in the V1 compartment (informational / upstream-MOSAIC artifact; not read by the model).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/`.
- **Related parameters**: [`V1_j_initial`](#v1_j_initial), [`prop_V2_initial`](#prop_v2_initial), [`prop_S_initial`](#prop_s_initial), [`prop_E_initial`](#prop_e_initial), [`prop_I_initial`](#prop_i_initial), [`prop_R_initial`](#prop_r_initial)
- **Notes**: No live consumer; converted to `float32` by `dict_to_propertysetex` but never read.

### `prop_V2_initial`

- **What it controls**: Per-patch initial fraction of population in the V2 compartment (informational / upstream-MOSAIC artifact; not read by the model).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: unconstrained (no shape or range assert in `validate_parameters`).
- **Off-value**: n/a (not a feature toggle).
- **Consumer code**: none in `src/laser/cholera/`.
- **Related parameters**: [`V2_j_initial`](#v2_j_initial), [`prop_V1_initial`](#prop_v1_initial), [`prop_S_initial`](#prop_s_initial), [`prop_E_initial`](#prop_e_initial), [`prop_I_initial`](#prop_i_initial), [`prop_R_initial`](#prop_r_initial)
- **Notes**: No live consumer; converted to `float32` by `dict_to_propertysetex` but never read.
