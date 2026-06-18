# Vaccination

The vaccination parameters configure the two-dose oral cholera vaccine (OCV) campaign layered on top of the SEIR core. Together they specify *when and where* doses are delivered (`nu_1_jt`, `nu_2_jt`), *how effective* each dose is at moving recipients into the vaccinated compartments (`phi_1`, `phi_2`), *how quickly* protection wanes back to susceptible (`omega_1`, `omega_2`), and *which compartments* first doses are drawn from (`nu_jt_sources`). The whole vaccination feature is gated by the dose-schedule arrays: setting `nu_1_jt` and `nu_2_jt` to all-zero arrays short-circuits both dose paths in `vaccinated.py`, leaving the effectiveness and waning parameters with nothing to act on.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
| --- | --- | --- | --- | --- |
| [`nu_1_jt`](#nu_1_jt) | `(nticks, npatches)` | `float32` | shape-only check | `np.zeros((nticks, npatches))` |
| [`nu_2_jt`](#nu_2_jt) | `(nticks, npatches)` | `float32` | shape-only check | `np.zeros((nticks, npatches))` |
| [`phi_1`](#phi_1) | `scalar` | `float32` | `[0, 1]` | n/a (not a feature toggle) |
| [`phi_2`](#phi_2) | `scalar` | `float32` | `[0, 1]` | n/a (not a feature toggle) |
| [`omega_1`](#omega_1) | `scalar` | `float32` | `>= 0` | `0` |
| [`omega_2`](#omega_2) | `scalar` | `float32` | `>= 0` | `0` |
| [`nu_jt_sources`](#nu_jt_sources) | `list[str]` | list of str | unconstrained | n/a (not a feature toggle) |

### `nu_1_jt`

- **What it controls**: Number of first-dose vaccinations to deliver per tick and per patch.
- **Shape**: `(nticks, npatches)`
- **Dtype**: `float32`
- **Range**: shape-only check; the validator asserts `params.nu_1_jt.shape == (nticks, npatches)` (the commented-out per-element `<= N_j_initial / 7` assert is disabled).
- **Off-value**: `np.zeros((nticks, npatches))` — `vaccinated.py` guards the dose-1 path with `if any(model.params.nu_1_jt[tick]):`, so an all-zero array short-circuits the entire first-dose delivery loop and no agents are moved into V1.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`nu_2_jt`](#nu_2_jt), [`phi_1`](#phi_1), [`omega_1`](#omega_1), [`nu_jt_sources`](#nu_jt_sources), [`V1_j_initial`](initial-populations.md#v1_j_initial)
- **Notes**: Ingestion transposes `(npatches, nticks)` input to `(nticks, npatches)` so the in-memory layout is tick-major. The off-value is the canonical "vaccination off" switch — all other vaccination parameters are inert without it.

### `nu_2_jt`

- **What it controls**: Number of second-dose vaccinations to deliver per tick and per patch (moves V1 to V2).
- **Shape**: `(nticks, npatches)`
- **Dtype**: `float32`
- **Range**: shape-only check; the validator asserts `params.nu_2_jt.shape == (nticks, npatches)` (the commented-out per-element `<= N_j_initial / 7` assert is disabled).
- **Off-value**: `np.zeros((nticks, npatches))` — `vaccinated.py` guards the dose-2 path with `if any(model.params.nu_2_jt[tick]):`, so an all-zero array short-circuits the V1->V2 transition and no agents are promoted from V1 to V2.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`nu_1_jt`](#nu_1_jt), [`phi_2`](#phi_2), [`omega_2`](#omega_2), [`V2_j_initial`](initial-populations.md#v2_j_initial)
- **Notes**: Ingestion transposes `(npatches, nticks)` input to `(nticks, npatches)`. Second doses can only meaningfully fire on patches and ticks where V1 has been built up by prior `nu_1_jt` delivery (or by a nonzero `V1_j_initial`).

### `phi_1`

- **What it controls**: Effectiveness (take fraction) of a first dose: only this fraction of delivered doses moves the recipient into V1.
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `[0, 1]`
- **Off-value**: `n/a (not a feature toggle)` — `phi_1` has no off-switch role; the dose-1 path is gated by `nu_1_jt`, not by `phi_1`. Setting `phi_1 = 0` would also zero out the effective-doses computation, but the canonical way to disable first-dose vaccination is to zero `nu_1_jt`.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`nu_1_jt`](#nu_1_jt), [`phi_2`](#phi_2), [`omega_1`](#omega_1), [`nu_jt_sources`](#nu_jt_sources)
- **Notes**: Plan §4 explicitly states `phi_1` has no effect once `nu_1_jt = 0`; the gate is the dose schedule, not the take fraction.

### `phi_2`

- **What it controls**: Effectiveness (take fraction) of a second dose: only this fraction of V1 recipients moves into V2.
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `[0, 1]`
- **Off-value**: `n/a (not a feature toggle)` — `phi_2` has no off-switch role; the dose-2 path is gated by `nu_2_jt`. Setting `phi_2 = 0` would also zero out the V1->V2 transition, but the canonical way to disable second-dose vaccination is to zero `nu_2_jt`.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`nu_2_jt`](#nu_2_jt), [`phi_1`](#phi_1), [`omega_2`](#omega_2)
- **Notes**: Plan §4 explicitly states `phi_2` has no effect once `nu_2_jt = 0`; the gate is the dose schedule, not the take fraction.

### `omega_1`

- **What it controls**: Daily waning rate from V1 back to S; converted to a per-tick probability via `1 - exp(-omega_1)`.
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `>= 0` (validator assert message says "omega_1 value must be positive", but the assert is `>= 0.0`, so `0` is allowed).
- **Off-value**: `0` — yields `self._omega_1_prob = -expm1(0) = 0`, so `binomial(V1_next, 0)` returns `0` and no V1 agents wane back to S.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`omega_2`](#omega_2), [`phi_1`](#phi_1), [`nu_1_jt`](#nu_1_jt), [`epsilon`](disease-progression.md#epsilon)
- **Notes**: The "must be positive" wording in the validator is a known inconsistency with the actual `>= 0.0` check; treat `0` as a valid (and operationally meaningful) value when waning is to be disabled.

### `omega_2`

- **What it controls**: Daily waning rate from V2 back to S; converted to a per-tick probability via `1 - exp(-omega_2)`.
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `>= 0` (validator assert message says "omega_2 value must be positive", but the assert is `>= 0.0`, so `0` is allowed).
- **Off-value**: `0` — yields `self._omega_2_prob = -expm1(0) = 0`, so no V2 agents wane back to S.
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/params.py`](../../reference/index.md)
- **Related parameters**: [`omega_1`](#omega_1), [`phi_2`](#phi_2), [`nu_2_jt`](#nu_2_jt), [`epsilon`](disease-progression.md#epsilon)
- **Notes**: The "must be positive" wording in the validator is a known inconsistency with the actual `>= 0.0` check; treat `0` as a valid (and operationally meaningful) value when waning is to be disabled.

### `nu_jt_sources`

- **What it controls**: List of compartment names from which first-dose vaccinations are drawn (proportional sampling across the listed compartments).
- **Shape**: `list[str]` (variable length).
- **Dtype**: list of `str` (not coerced).
- **Range**: unconstrained (not asserted by `validate_parameters`). This is **not free text** — it is a structural list of source-compartment labels. Recognized labels are `"S"`, `"E"`, `"Isym"`, `"Iasym"`, and `"R"`; any other entry is silently dropped because `vaccinated.py` filters the list with `hasattr(model.people, name)`.
- **Off-value**: `n/a (not a feature toggle)` — the way to disable first-dose vaccination is to zero `nu_1_jt`, not to empty this list. (An empty list would also disable dose-1, but is not the canonical off-switch.)
- **Consumer code**: [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md), [`src/laser/cholera/metapop/utils.py`](../../reference/index.md)
- **Related parameters**: [`nu_1_jt`](#nu_1_jt), [`phi_1`](#phi_1)
- **Notes**: Optional. If absent from the parameter set, `vaccinated.py` defaults to `["S", "E", "Isym", "Iasym", "R"]`, and the bundled `default_parameters.json` ships with the same five-compartment list. Restricting the list (for example to `["S"]`) changes the dynamics: dose-1 then only draws from susceptibles and never "wastes" doses on already-exposed or recovered agents. See plan §4 for the dynamic consequences of narrowing this list.
