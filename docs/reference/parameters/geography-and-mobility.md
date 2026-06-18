# Geography & mobility

This group defines where each patch sits on the globe and how strongly its inhabitants mix with the other patches. The two coordinate vectors (`longitude`, `latitude`) feed the great-circle distance kernel that — together with the gravity exponents `mobility_omega` and `mobility_gamma` — builds the per-patch coupling matrix `pi_ij`. The per-patch coupling strength `tau_i` is the master switch: it scales how much of a patch's infectious (and, for the environmental pathway, susceptible) fraction is treated as moving out of its home patch each tick. Set `tau_i` to zeros and the spatial-mixing terms drop out of the dynamics; the other four parameters still have to be present for validation to pass, but they no longer influence the run.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
| --- | --- | --- | --- | --- |
| [`longitude`](#longitude) | `(npatches,)` | `float32` | shape-only check | n/a |
| [`latitude`](#latitude) | `(npatches,)` | `float32` | shape-only check | n/a |
| [`tau_i`](#tau_i) | `(npatches,)` | `float32` | `[0, 1]` per element | `np.zeros(npatches)` |
| [`mobility_omega`](#mobility_omega) | scalar | `float32` | unconstrained | n/a |
| [`mobility_gamma`](#mobility_gamma) | scalar | `float32` | unconstrained | n/a |

### `longitude`

- **What it controls**: Per-patch longitude coordinate; used together with `latitude` to compute great-circle distances that feed the gravity-model mobility matrix.
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: shape-only check — `validate_parameters` asserts `len(params.longitude) == npatches` and nothing else.
- **Off-value**: `n/a` — `longitude` is geographic input, not a feature toggle; it cannot be "switched off". The mobility dynamics are disabled by zeroing [`tau_i`](#tau_i), not by zeroing the coordinates.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`latitude`](#latitude), [`tau_i`](#tau_i), [`mobility_omega`](#mobility_omega), [`mobility_gamma`](#mobility_gamma)
- **Notes**: Read inside `utils.py`'s `distance(...)` helper when assembling `pi_ij`. When `tau_i = 0` the coordinates are still required by the validator, but the resulting `pi_ij` is never multiplied into a non-zero term so the values do not influence the dynamics.

### `latitude`

- **What it controls**: Per-patch latitude coordinate; combined with `longitude` for inter-patch distance in the gravity-mobility kernel (and used to order patches in one plot).
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: shape-only check — `validate_parameters` asserts `len(params.latitude) == npatches` and nothing else.
- **Off-value**: `n/a` — geographic input, not a feature toggle. See [`tau_i`](#tau_i) for the inert setting that removes spatial mixing from the dynamics.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md)
- **Related parameters**: [`longitude`](#longitude), [`tau_i`](#tau_i), [`mobility_omega`](#mobility_omega), [`mobility_gamma`](#mobility_gamma)
- **Notes**: `humantohuman.py` uses `latitude` only to sort patches for a diagnostic plot; the dynamics consume it only through the `distance(...)` call in `utils.py`.

### `tau_i`

- **What it controls**: Per-patch fraction of infectious (and, for environmental exposure, susceptible) individuals that move out of their home patch; the master switch for spatial mixing.
- **Shape**: `(npatches,)`
- **Dtype**: `float32`
- **Range**: `[0, 1]` per element — `validate_parameters` asserts `np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0))`. The same range assertion is repeated in `dict_to_propertysetex` (line 554) and `validate_parameters` (line 746) in `params.py`.
- **Off-value**: `np.zeros(npatches)` — with `tau_i = 0`, `humantohuman.py` sets `local_frac = 1` and the immigrating term `(0 * total_i) * pi_ij.T` collapses to zero, so the force of infection uses only the local infected count and `pi_ij` drops out. In `envtohuman.py` `local_frac = 1 - tau_i = 1`, so every susceptible is exposed locally and patches do not exchange environmental exposure.
- **Consumer code**: [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/envtohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/derivedvalues.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`mobility_omega`](#mobility_omega), [`mobility_gamma`](#mobility_gamma), [`longitude`](#longitude), [`latitude`](#latitude), `pi_ij`
- **Notes**: `tau_i` is the only parameter in this group with a validator-enforced numeric range; the coordinates and the gravity exponents are validated for presence/shape only. Treat `tau_i` as the canonical knob for turning spatial mixing on or off — do not try to disable mobility by, for example, setting `mobility_omega = 0` or zeroing the coordinates.

### `mobility_omega`

- **What it controls**: Exponent on the destination population in the gravity model that builds the patch-to-patch coupling matrix `pi_ij`; larger values pull more flow toward larger patches.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: unconstrained — `validate_parameters` only checks `"mobility_omega" in params`. Any float passes validation.
- **Off-value**: `n/a` — `mobility_omega` is a kernel shape parameter, not a feature toggle. **It is validator-required even when [`tau_i`](#tau_i) is zero**: `pi_ij` is still constructed inside `utils.py` using `mobility_omega`, but every term it contributes to is multiplied by `tau_i = 0` in `humantohuman.py` and so it has no effect on the dynamics. The right way to disable spatial mixing is to set [`tau_i`](#tau_i) to zeros while leaving `mobility_omega` at any valid float.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`mobility_gamma`](#mobility_gamma), [`tau_i`](#tau_i), [`longitude`](#longitude), [`latitude`](#latitude)

### `mobility_gamma`

- **What it controls**: Exponent on inter-patch distance (with sign reversed) in the gravity model; larger values penalize long-distance movement more strongly when building `pi_ij`.
- **Shape**: scalar
- **Dtype**: `float32`
- **Range**: unconstrained — `validate_parameters` only checks `"mobility_gamma" in params`. Any float passes validation.
- **Off-value**: `n/a` — kernel shape parameter, not a feature toggle. **It is validator-required even when [`tau_i`](#tau_i) is zero**: `pi_ij` is still assembled with `mobility_gamma` in the distance term, but the resulting matrix is zeroed out of the dynamics by the `tau_i = 0` multiplication in `humantohuman.py`. Disable spatial mixing by zeroing [`tau_i`](#tau_i), not by adjusting `mobility_gamma`.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`mobility_omega`](#mobility_omega), [`tau_i`](#tau_i), [`longitude`](#longitude), [`latitude`](#latitude)
