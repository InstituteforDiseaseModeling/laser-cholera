# Run identity & calendar

This group fixes the four scalars and labels that every other parameter is interpreted against: the PRNG seed (and therefore which stochastic realisation you get), the calendar window `[date_start, date_stop]` from which the derived attribute `nticks` is computed, and the ordered `location_name` list whose length defines `npatches` and whose order defines the element ordering of every per-patch vector in the rest of the configuration. Together they answer the questions "which run is this?", "how long does it run?", and "which patches are we talking about, and in what order?". Nothing else in the parameter set has meaning until these four are settled.

## Quick reference

| Parameter | Shape | Dtype | Range | Off-value |
|---|---|---|---|---|
| [`seed`](#seed) | scalar | int (untyped) | unconstrained | n/a (not a toggle) |
| [`date_start`](#date_start) | scalar | `datetime.datetime` | `date_stop >= date_start` | n/a (not a toggle) |
| [`date_stop`](#date_stop) | scalar | `datetime.datetime` | `date_stop >= date_start` | n/a (not a toggle) |
| [`location_name`](#location_name) | `list[str]` of length `npatches` | list of str | shape-only check (every per-location vector must match its length) | n/a (not a toggle) |

### `seed`

- **What it controls**: Integer seed for the model's pseudo-random number generator; setting it pins all stochastic draws to a reproducible sequence.
- **Shape**: `scalar`
- **Dtype**: `int (untyped; passed through as-is)`
- **Range**: unconstrained (the validator does not check it; `None` falls back to the current microsecond at model construction)
- **Off-value**: `n/a` — `seed` is not a feature toggle. To intentionally request a non-reproducible run, pass `None` and the model uses `self.tinit.microsecond` instead.
- **Consumer code**: [`src/laser/cholera/metapop/model.py`](../../reference/index.md)
- **Related parameters**: none
- **Notes**: `params.py` explicitly comments "No processing of 'params.seed'" — the field is not coerced to a numpy dtype and is not range-checked by `validate_parameters`. The sole consumer is `model.py` which calls `seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)`, so `None` is the supported way to opt out of reproducibility.

### `date_start`

- **What it controls**: Calendar date of simulation tick 0; together with `date_stop` it defines the run length (`nticks = (date_stop - date_start).days + 1`).
- **Shape**: `scalar`
- **Dtype**: `datetime.datetime` (parsed from a `"%Y-%m-%d"` string; passed through if already a `datetime`)
- **Range**: must satisfy `date_stop >= date_start` (asserted in `validate_parameters`)
- **Off-value**: `n/a` — `date_start` is not a feature toggle; every run has a start date.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/analyzer.py`](../../reference/index.md), [`src/laser/cholera/calc_model_likelihood.py`](../../reference/index.md)
- **Related parameters**: [`date_stop`](#date_stop), [`epidemic_peaks`](regime-switching.md#epidemic_peaks)
- **Notes**: The derived attribute `params.nticks` is computed from this pair at ingestion time, so mutating `date_start` after `dict_to_propertysetex` runs would desynchronise `nticks`. `analyzer.py` forwards `date_start` / `date_stop` into `calc_model_likelihood` for the in-window `epidemic_peaks` filter.

### `date_stop`

- **What it controls**: Calendar date of the final simulation tick; with `date_start` it bounds the run window and sets `nticks`.
- **Shape**: `scalar`
- **Dtype**: `datetime.datetime` (parsed from a `"%Y-%m-%d"` string; passed through if already a `datetime`)
- **Range**: `date_stop >= date_start` (asserted by `validate_parameters` with the message `"date_stop ({params.date_stop}) must be >= date_start ({params.date_start})"`)
- **Off-value**: `n/a` — `date_stop` is not a feature toggle; every run has an end date.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/analyzer.py`](../../reference/index.md), [`src/laser/cholera/calc_model_likelihood.py`](../../reference/index.md)
- **Related parameters**: [`date_start`](#date_start), [`epidemic_peaks`](regime-switching.md#epidemic_peaks)
- **Notes**: Setting `date_stop == date_start` yields `nticks == 1` (a single-tick run), which is the smallest valid window. Like `date_start`, this is read by `analyzer.py` and forwarded into `calc_model_likelihood` to drop `epidemic_peaks` rows whose `peak_date` falls outside `[date_start, date_stop]`.

### `location_name`

- **What it controls**: Ordered list of patch / admin-unit labels; its length defines `npatches` and its order defines every per-patch vector's element ordering.
- **Shape**: `list[str]` of length `npatches` (a scalar input is promoted to a single-element list)
- **Dtype**: list of `str` (not coerced to `ndarray`)
- **Range**: shape-only check — every per-location vector must have `len(location_name)` entries; if `epidemic_peaks` is present, every `iso_code` in it must appear in `location_name` (asserted in `dict_to_propertysetex`)
- **Off-value**: `n/a` — `location_name` is not a feature toggle; a run must name its patches.
- **Consumer code**: [`src/laser/cholera/metapop/params.py`](../../reference/index.md), [`src/laser/cholera/metapop/census.py`](../../reference/index.md), [`src/laser/cholera/metapop/derivedvalues.py`](../../reference/index.md), [`src/laser/cholera/metapop/environmental.py`](../../reference/index.md), [`src/laser/cholera/metapop/envtohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/exposed.py`](../../reference/index.md), [`src/laser/cholera/metapop/humantohuman.py`](../../reference/index.md), [`src/laser/cholera/metapop/infectious.py`](../../reference/index.md), [`src/laser/cholera/metapop/recovered.py`](../../reference/index.md), [`src/laser/cholera/metapop/susceptible.py`](../../reference/index.md), [`src/laser/cholera/metapop/vaccinated.py`](../../reference/index.md)
- **Related parameters**: [`longitude`](geography-and-mobility.md#longitude), [`latitude`](geography-and-mobility.md#latitude), [`N_j_initial`](initial-populations.md#n_j_initial), [`epidemic_peaks`](regime-switching.md#epidemic_peaks)
- **Notes**: `params.py` promotes a scalar value to a single-element list (so a one-patch run can pass `"COD"` rather than `["COD"]`). `dict_to_propertysetex` also rewrites `epidemic_peaks` by adding a `loc_idx` column derived from `location_name`'s order, so changing the order after ingestion would silently mis-align peak locations. Almost every distribution module reads `location_name` only for plot labels — it has no dynamical effect beyond setting `npatches`.
