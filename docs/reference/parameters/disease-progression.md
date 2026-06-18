# Disease progression

These parameters control how individuals move through the SEIR portion of the compartmental model once they have been infected: how fast exposed people become infectious (`iota`), how new infections split between symptomatic and asymptomatic cases (`sigma`), how quickly each infectious branch recovers (`gamma_1`, `gamma_2`), and how fast natural immunity wanes back to susceptibility (`epsilon`). Together they set the per-tick transition probabilities used by `infectious.py`, `exposed.py`, and `recovered.py`.

All five are scalars stored as `float32`, and each is consumed inside the model components via the `1 - exp(-rate)` transformation to convert a per-day rate into a per-tick Bernoulli probability for a `Binomial` draw.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
|---|---|---|---|---|
| [`iota`](#iota) | `scalar` | `float32` | `>= 0` | `0` |
| [`gamma_1`](#gamma_1) | `scalar` | `float32` | `>= 0` | `0` |
| [`gamma_2`](#gamma_2) | `scalar` | `float32` | `>= 0` | `0` |
| [`epsilon`](#epsilon) | `scalar` | `float32` | `>= 0` | `0` |
| [`sigma`](#sigma) | `scalar` | `float32` | `[0, 1]` | `1` |

### `iota`

- **What it controls**: Per-day rate at which exposed (E) individuals progress to becoming infectious (I_sym or I_asym).
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `>= 0` (validator-enforced; the assert message reads `"iota value must be positive"` but `0` is allowed).
- **Off-value**: `0` — sets the per-tick progression probability `1 - exp(-iota)` to zero, freezing the exposed cohort (apart from non-disease deaths) so no new infectious individuals are produced.
- **Consumer code**: [`src/laser/cholera/metapop/exposed.py`](../../reference/index.md), [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`sigma`](#sigma), [`gamma_1`](#gamma_1), [`gamma_2`](#gamma_2)
- **Notes**: Consumed in `infectious.py` as `_iota_prob = 1 - exp(-iota)`, then `progressing = Binomial(E_next, _iota_prob)`. The validator wording `"must be positive"` is slightly inconsistent with the underlying `>= 0` assert — `0` passes validation.

### `gamma_1`

- **What it controls**: Per-day recovery rate for symptomatic infectious individuals (I_sym -> R).
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `>= 0` (validator-enforced; the assert message reads `"gamma_1 value must be positive"` but `0` is allowed).
- **Off-value**: `0` — disables symptomatic recovery; symptomatic infectious individuals never transition to R via this pathway (they can still die through vital-dynamics mortality).
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`gamma_2`](#gamma_2), [`iota`](#iota), [`sigma`](#sigma)
- **Notes**: Cached as `_gamma_1_prob = 1 - exp(-gamma_1)` in `infectious.check()` and used as the Bernoulli probability for each symptomatic individual's recovery draw. Not listed in the plan §4 inert-values table because disabling recovery is not a feature you would normally toggle off, but it works mathematically.

### `gamma_2`

- **What it controls**: Per-day recovery rate for asymptomatic infectious individuals (I_asym -> R).
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `>= 0` (validator-enforced; the assert message reads `"gamma_2 value must be positive"` but `0` is allowed).
- **Off-value**: `0` — disables asymptomatic recovery; asymptomatic infectious individuals never transition to R via this pathway.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`gamma_1`](#gamma_1), [`iota`](#iota), [`sigma`](#sigma)
- **Notes**: Cached as `_gamma_2_prob = 1 - exp(-gamma_2)` and used as the Bernoulli probability for asymptomatic recovery. Not listed in the plan §4 inert-values table.

### `epsilon`

- **What it controls**: Per-day rate at which natural immunity wanes (R -> S).
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `>= 0` (validator-enforced; the assert message reads `"epsilon value must be positive"` but `0` is allowed).
- **Off-value**: `0` — disables natural-immunity waning; the recovered compartment is monotone (apart from natural mortality) and individuals never return to S via this pathway.
- **Consumer code**: [`src/laser/cholera/metapop/recovered.py`](../../reference/index.md)
- **Related parameters**: [`omega_1`](vaccination.md#omega_1), [`omega_2`](vaccination.md#omega_2)
- **Notes**: Consumed as `_waning_prob = 1 - exp(-epsilon)`, then `waned = Binomial(R - non_disease_deaths, _waning_prob)`. Matches the plan §4 inert-values row: `epsilon = 0` -> R -> S waning disabled.

### `sigma`

- **What it controls**: Fraction of new infections that become symptomatic (the remainder are routed to the asymptomatic compartment).
- **Shape**: `scalar`
- **Dtype**: `float32`
- **Range**: `[0, 1]` (validator-enforced; assert message `"sigma value must be in the range [0, 1]"`).
- **Off-value**: `1` — routes 100% of new infections to I_sym, leaving I_asym empty; collapses the symptomatic/asymptomatic split to a single branch while preserving the case-reporting pathway (cases are scored against I_sym only). The complementary value `sigma = 0` also collapses the split — routing everything to I_asym — but yields zero reported cases even when infections occur, so `1` is the canonical off-value for this docs convention.
- **Consumer code**: [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md)
- **Related parameters**: [`iota`](#iota), [`gamma_1`](#gamma_1), [`gamma_2`](#gamma_2)
- **Notes**: Used in two places in `infectious.py`: (1) initial seeding `Isym[0] = round(sigma * I_j_initial); Iasym[0] = I_j_initial - Isym[0]`, and (2) per-tick split of new progressors `new_symptomatic = round(sigma * progressing); new_asymptomatic = progressing - new_symptomatic`. Either extreme (`0` or `1`) mathematically collapses the split; choosing `1` as the documented off-value preserves observability through the case-reporting channel.
