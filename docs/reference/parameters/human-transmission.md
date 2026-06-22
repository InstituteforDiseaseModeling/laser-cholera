# Human-to-human transmission

This group sets the direct, person-to-person transmission term of the cholera model: a per-patch baseline rate, the shape of its seasonal envelope, and the mixing nonlinearity that turns infected counts into a force of infection. Together these parameters control whether S to E transitions happen at all (via `beta_j0_hum`), how that rate breathes with the calendar (`a_1_j`, `b_1_j`, `a_2_j`, `b_2_j`, `p`), and how the FOI scales with infected prevalence and patch population (`alpha_1`, `alpha_2`). Two legacy fields (`beta_j0_tot`, `p_beta`) are parsed for round-trip compatibility with upstream MOSAIC R configs but are not read anywhere in the Python simulation loop.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
| --- | --- | --- | --- | --- |
| [`beta_j0_hum`](#beta_j0_hum) | `(npatches,)` | `np.float32` | `>= 0` (per-element) | `np.zeros(npatches)` |
| [`a_1_j`](#a_1_j) | `(npatches,)` | `np.float32` | unconstrained | `np.zeros(npatches)` |
| [`a_2_j`](#a_2_j) | `(npatches,)` | `np.float32` | unconstrained | `np.zeros(npatches)` |
| [`b_1_j`](#b_1_j) | `(npatches,)` | `np.float32` | unconstrained | `np.zeros(npatches)` |
| [`b_2_j`](#b_2_j) | `(npatches,)` | `np.float32` | unconstrained | `np.zeros(npatches)` |
| [`p`](#p) | scalar | `np.int32` | integral; `"p" in params` | `365` (any non-zero) |
| [`alpha_1`](#alpha_1) | scalar **or** `(npatches,)` | `np.float32` | scalar or per-element in `(0, 1]` | n/a (not a toggle) |
| [`alpha_2`](#alpha_2) | scalar **or** `(npatches,)` | `np.float32` | scalar or per-element in `[0, 1]` | n/a (not a toggle) |
| [`beta_j0_tot`](#beta_j0_tot) | `(npatches,)` | `np.float32` | unconstrained | n/a (inert / unused) |
| [`p_beta`](#p_beta) | `(npatches,)` | `np.float32` | unconstrained | n/a (inert / unused) |

## Parameters

### `beta_j0_hum`

- **What it controls**: Per-patch baseline human-to-human transmission rate; multiplied by the seasonal harmonic to produce the per-tick, per-patch transmission envelope.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: `>= 0` per element; the validator asserts `np.all(params.beta_j0_hum >= 0.0)` and `len(params.beta_j0_hum) == npatches`.
- **Off-value**: `np.zeros(npatches)` — zeroes the seasonal envelope `beta_jt_human` so `Lambda` collapses to 0 and no S to E transitions occur, disabling the human-to-human term entirely.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/derivedvalues.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`a_1_j`](#a_1_j), [`b_1_j`](#b_1_j), [`a_2_j`](#a_2_j), [`b_2_j`](#b_2_j), [`p`](#p), [`alpha_1`](#alpha_1), [`alpha_2`](#alpha_2), [`tau_i`](geography-and-mobility.md#tau_i), [`beta_j0_env`](environmental-transmission.md#beta_j0_env)
- **Notes**: Raw `beta_j0_hum` is only consumed by `get_daily_seasonality` (which builds `patches.beta_jt_human`) and by the validator; `humantohuman.py` and `derivedvalues.py` reach it indirectly through `patches.beta_jt_human`. Because the whole harmonic envelope is multiplied by `beta_j0_hum`, zeroing this vector is the canonical way to switch the human-to-human pathway off without touching the seasonality coefficients.

### `a_1_j`

- **What it controls**: Per-patch amplitude of the first (annual) cosine term of the seasonal-transmission harmonic.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained (the validator only checks `len(params.a_1_j) == npatches`).
- **Off-value**: `np.zeros(npatches)` — removes the annual cosine contribution; combined with `a_2_j = b_1_j = b_2_j = 0` the seasonal bracket collapses to a constant `1.0`, leaving a flat (non-seasonal) envelope at `beta_j0_hum`.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`b_1_j`](#b_1_j), [`a_2_j`](#a_2_j), [`b_2_j`](#b_2_j), [`p`](#p), [`beta_j0_hum`](#beta_j0_hum)
- **Notes**: In `get_daily_seasonality` (`utils.py:126-135`) the bracket `1 + a1*cos(2*pi*t/p) + b1*sin(2*pi*t/p) + a2*cos(4*pi*t/p) + b2*sin(4*pi*t/p)` multiplies `beta_j0_hum`. Zeroing the four amplitudes is the safe way to disable seasonality; do not touch the period `p`, which is the denominator.

### `a_2_j`

- **What it controls**: Per-patch amplitude of the second (semi-annual) cosine term of the seasonal-transmission harmonic.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained (the validator only checks `len(params.a_2_j) == npatches`).
- **Off-value**: `np.zeros(npatches)` — removes the semi-annual cosine contribution; combined with `a_1_j = b_1_j = b_2_j = 0` the seasonal bracket collapses to `1.0`.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`a_1_j`](#a_1_j), [`b_1_j`](#b_1_j), [`b_2_j`](#b_2_j), [`p`](#p), [`beta_j0_hum`](#beta_j0_hum)
- **Notes**: Multiplies `cos(4*pi*t/p)` in `utils.py:132`. Like `a_1_j`, this is an amplitude (not a period); zeroing it is safe and does not produce a divide-by-zero.

### `b_1_j`

- **What it controls**: Per-patch amplitude of the first (annual) sine term of the seasonal-transmission harmonic.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained (the validator only checks `len(params.b_1_j) == npatches`).
- **Off-value**: **Verified off-value (corrects plan §4):** `np.zeros(npatches)`. Plan §4 (line 70) suggests `b_1_j = b_2_j = 1` and warns that zeroing them would divide by zero. That is incorrect: in `get_daily_seasonality` (`utils.py:126-135`), `b_1_j` and `b_2_j` are *sine amplitudes* multiplying `sin(2*pi*t/p)` and `sin(4*pi*t/p)` — they never appear in a denominator. The actual divide-by-zero hazard is the period `p`. Zeroing `b_1_j` alongside the other three amplitudes cleanly collapses the harmonic bracket to `1.0`.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`a_1_j`](#a_1_j), [`a_2_j`](#a_2_j), [`b_2_j`](#b_2_j), [`p`](#p), [`beta_j0_hum`](#beta_j0_hum)
- **Notes**: The off-trick for seasonality is to zero the amplitudes (`a_1_j`, `a_2_j`, `b_1_j`, `b_2_j`) and leave the period [`p`](#p) alone. Equivalently, setting `beta_j0_hum = 0` zeroes the whole envelope regardless of these coefficients.

### `b_2_j`

- **What it controls**: Per-patch amplitude of the second (semi-annual) sine term of the seasonal-transmission harmonic.
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained (the validator only checks `len(params.b_2_j) == npatches`).
- **Off-value**: **Verified off-value (corrects plan §4):** `np.zeros(npatches)`. Same correction as `b_1_j`: plan §4 mislabels `b_2_j` as a period that must be non-zero, but it is the sine amplitude on `sin(4*pi*t/p)` and never divides. Setting it to `0` together with `a_1_j = a_2_j = b_1_j = 0` collapses the harmonic bracket to `1.0` with no divide-by-zero risk.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`a_1_j`](#a_1_j), [`a_2_j`](#a_2_j), [`b_1_j`](#b_1_j), [`p`](#p), [`beta_j0_hum`](#beta_j0_hum)
- **Notes**: The off-trick for seasonality is to zero the amplitudes (`a_*`, `b_*`), never the period. The actual divide-by-zero hazard in this expression is [`p`](#p).

### `p`

- **What it controls**: Period (in ticks/days) of the seasonal-transmission harmonic; typically `365` for an annual cycle.
- **Shape**: scalar
- **Dtype**: `np.int32`
- **Range**: validator asserts `"p" in params` and the ingestion path asserts `int(params.p) == params.p` (must be integral). There is no explicit positivity check in `validate_parameters`, but `p` appears as the denominator of `2*pi*t/p` and `4*pi*t/p` in `utils.py:130-133`, so it must be non-zero.
- **Off-value**: `365` (or any non-zero integer). `p` is *not* the feature toggle for seasonality — zeroing the amplitudes (`a_1_j`, `a_2_j`, `b_1_j`, `b_2_j`) is. Leave `p` at its default and disable seasonality through the amplitudes.
- **Consumer code**: [`src/laser/cholera/metapop/utils.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/derivedvalues.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`a_1_j`](#a_1_j), [`a_2_j`](#a_2_j), [`b_1_j`](#b_1_j), [`b_2_j`](#b_2_j), [`beta_j0_hum`](#beta_j0_hum)
- **Notes**: `derivedvalues.py.check()` asserts `"p" in params` even though `derivedvalues.py` itself never reads `p`; the check is a defensive consistency check inherited from when the spatial-hazard math used it directly.

### `alpha_1`

- **What it controls**: Exponent on the effective-infected count in the human-to-human force of infection (mixing nonlinearity in the numerator).
- **Shape**: scalar **or** `(npatches,)` array.
- **Dtype**: `np.float32`
- **Range**: scalar or per-element in `(0, 1]` — the validator asserts strict-positive at the low end. The lower bound is strict because `alpha_1 = 0` collapses `np.power(effective_i, 0)` to `1`, breaking the FOI's dependence on infected counts.
- **Off-value**: n/a — `alpha_1` is a mixing-shape parameter, not a feature toggle. To turn the human-to-human term off, set [`beta_j0_hum`](#beta_j0_hum) to zero; the validator forbids `alpha_1 = 0`, which would collapse the I-dependence to a constant.
- **Consumer code**: [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`alpha_2`](#alpha_2), [`beta_j0_hum`](#beta_j0_hum), [`tau_i`](geography-and-mobility.md#tau_i)
- **Notes**: Applied as `np.power(effective_i, alpha_1)` in `humantohuman.py`, where `effective_i` is the mobility-weighted infected count. `np.power` broadcasts cleanly over either shape — pass a scalar when every patch should share the same mixing nonlinearity, or a length-`npatches` array when the nonlinearity is heterogeneous across admin units. Ingestion length-checks the array against `npatches`; a wrong-length array is rejected at `get_parameters` time, not at `model.run()`.

### `alpha_2`

- **What it controls**: Exponent on the patch population in the FOI denominator: `alpha_2 = 1` gives frequency-dependent mixing, `alpha_2 = 0` gives density-dependent mixing.
- **Shape**: scalar **or** `(npatches,)` array.
- **Dtype**: `np.float32`
- **Range**: scalar or per-element in `[0, 1]`.
- **Off-value**: n/a — not a feature toggle; this parameter selects between mixing regimes. Disable the human-to-human term via [`beta_j0_hum`](#beta_j0_hum) instead.
- **Consumer code**: [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`alpha_1`](#alpha_1), [`beta_j0_hum`](#beta_j0_hum)
- **Notes**: Applied as `np.power(N, alpha_2)` in `humantohuman.py`. Same dual-mode acceptance as [`alpha_1`](#alpha_1): scalar for a globally-uniform mixing regime, or a length-`npatches` array to mix regimes across patches (e.g. dense urban districts at frequency-dependent, sparse rural districts at density-dependent).

### `beta_j0_tot`

- **What it controls**: Per-patch combined baseline transmission rate (legacy / metadata field carried from upstream MOSAIC R configs; not consumed by the Python simulation loop).
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained (no validator assert).
- **Off-value**: n/a — inert. Changing `beta_j0_tot` has no effect on dynamics; configure the human-to-human and environmental pathways via [`beta_j0_hum`](#beta_j0_hum) and [`beta_j0_env`](environmental-transmission.md#beta_j0_env) directly.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`beta_j0_hum`](#beta_j0_hum), [`beta_j0_env`](environmental-transmission.md#beta_j0_env), [`p_beta`](#p_beta)
- **Notes**: Coerced to an `np.float32` ndarray by `dict_to_propertysetex` (`params.py:517`) and listed in `override_helper`'s CLI-unsupported set (`utils.py:346`), but no module in `src/laser/cholera/` ever reads `params.beta_j0_tot`. It is preserved for round-trip JSON fidelity with upstream configs. Do not expect it to affect simulation results.

### `p_beta`

- **What it controls**: Per-patch split fraction between human-to-human and environmental transmission in the upstream R model (legacy metadata; not consumed by the Python simulation loop).
- **Shape**: `(npatches,)`
- **Dtype**: `np.float32`
- **Range**: unconstrained (no validator assert).
- **Off-value**: n/a — inert. The Python simulation uses `beta_j0_hum` and `beta_j0_env` independently rather than deriving them from `beta_j0_tot * p_beta`, so `p_beta` does not influence dynamics.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`beta_j0_hum`](#beta_j0_hum), [`beta_j0_env`](environmental-transmission.md#beta_j0_env), [`beta_j0_tot`](#beta_j0_tot)
- **Notes**: Coerced to an `np.float32` ndarray and listed in `override_helper`, but unread by the simulation. Like `beta_j0_tot`, `p_beta` is parsed for round-trip compatibility with upstream MOSAIC configs.
