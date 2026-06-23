# Enable vaccination

This guide turns on the two-dose oral cholera vaccine (OCV) layer on top of an otherwise-working parameter set. You will build the per-tick-per-patch dose schedules `nu_1_jt` and `nu_2_jt`, choose dose effectiveness (`phi_1`, `phi_2`), pick waning rates (`omega_1`, `omega_2`), and decide which compartments first doses are drawn from (`nu_jt_sources`). It assumes you already have a config that runs end-to-end with vaccination off; see [Override parameters](override-parameters.md) for the `mods` dict mechanics and [tutorials/single-location.md](../tutorials/single-location.md) Step 5b for an end-to-end walk-through.

## Prerequisites

- `laser-cholera` installed in your active environment.
- A working parameter set (e.g., the bundled `default_parameters.json`) that already runs with vaccination off (`nu_1_jt` and `nu_2_jt` all zero).
- Known `nticks` (number of simulation ticks) and `npatches` (`len(location_name)`) for that parameter set.
- Familiarity with the override pattern: a `mods` dict passed to the model constructor or merged into the parameter JSON.

## Steps

1. Build the first-dose schedule `nu_1_jt` with shape `(nticks, npatches)`. The example below delivers 50 doses per day in every patch starting at tick 30:

    ```python
    import numpy as np

    nu_1_jt = np.zeros((nticks, npatches), dtype=np.float32)
    nu_1_jt[30:, :] = 50.0
    ```

2. Set `phi_1`, the first-dose take fraction. Typical OCV first-dose effectiveness sits in `0.6`–`0.9`:

    ```python
    phi_1 = 0.75
    ```

3. Pick `omega_1`, the daily waning rate from V1 back to S. Use `0` to disable waning, or convert a half-life to a rate. A 5-year half-life is:

    ```python
    omega_1 = 0.0                          # no waning
    omega_1 = np.log(2) / (5 * 365.25)     # 5-year half-life
    ```

4. Repeat for the second dose. `nu_2_jt` has the same `(nticks, npatches)` shape; second doses can only fire on ticks where V1 has built up:

    ```python
    nu_2_jt = np.zeros((nticks, npatches), dtype=np.float32)
    nu_2_jt[60:, :] = 50.0
    phi_2 = 0.85
    omega_2 = np.log(2) / (5 * 365.25)
    ```

5. Set `nu_jt_sources` — the list of compartment labels from which first doses are drawn. Recognized labels are `"S"`, `"E"`, `"Isym"`, `"Iasym"`, `"R"`; any other string is silently dropped. The bundled default is all five:

    ```python
    nu_jt_sources = ["S", "E", "Isym", "Iasym", "R"]
    ```

## Worked example: `["S"]`-only vs. the bundled default

The source list controls who gets doses, and therefore how many doses become operationally useful. If your campaign realistically reaches only susceptibles, restrict the source list to `["S"]`; if doses are distributed broadly across the population, leave the default.

```python
mods_default = {
    "nu_1_jt": nu_1_jt,
    "nu_2_jt": nu_2_jt,
    "phi_1": 0.75,
    "phi_2": 0.85,
    "omega_1": np.log(2) / (5 * 365.25),
    "omega_2": np.log(2) / (5 * 365.25),
    "nu_jt_sources": ["S", "E", "Isym", "Iasym", "R"],
}

mods_susceptible_only = {
    "nu_1_jt": nu_1_jt,
    "nu_2_jt": nu_2_jt,
    "phi_1": 0.75,
    "phi_2": 0.85,
    "omega_1": np.log(2) / (5 * 365.25),
    "omega_2": np.log(2) / (5 * 365.25),
    "nu_jt_sources": ["S"],
}
```

The dynamic consequence: with `["S"]`, every delivered dose has a chance of moving a true-susceptible into V1; V1 climbs faster per delivered dose. With the bundled default, dose-1 draws proportionally from S, E, Isym, Iasym, and R, so a fraction of doses lands on already-exposed or already-recovered individuals and is operationally wasted, and V1 climbs more slowly per delivered dose for the same `nu_1_jt`.

## Full example

```python
import numpy as np

from laser.cholera.metapop.model import run_model
from laser.cholera.metapop.params import get_parameters

# Use `run_model` for the full default component pipeline; it forwards kwargs
# as `mods` to `get_parameters`, so any per-tick array we build here lands on
# the params set before validation runs.
defaults = get_parameters(None, do_validation=False)
nticks = defaults.nticks
npatches = len(defaults.location_name)

nu_1_jt = np.zeros((nticks, npatches), dtype=np.float32)
nu_1_jt[30:, :] = 50.0

nu_2_jt = np.zeros((nticks, npatches), dtype=np.float32)
nu_2_jt[60:, :] = 50.0

model = run_model(
    None,
    nu_1_jt=nu_1_jt,
    nu_2_jt=nu_2_jt,
    phi_1=0.75,
    phi_2=0.85,
    omega_1=float(np.log(2) / (5 * 365.25)),
    omega_2=float(np.log(2) / (5 * 365.25)),
    nu_jt_sources=["S", "E", "Isym", "Iasym", "R"],
)
```

## See also

- [Vaccination parameter reference](../reference/parameters/vaccination.md) — full per-parameter detail, shapes, ranges, and off-values.
- [tutorials/single-location.md](../tutorials/single-location.md) — Step 5b turns vaccination on in the single-location tutorial.
- [configurations/multi-admin.md](../configurations/multi-admin.md) — vaccination off; demonstrates the canonical off-form (`nu_1_jt = nu_2_jt = 0`).
- [Override parameters](override-parameters.md) — the `mods` dict mechanics used above.
