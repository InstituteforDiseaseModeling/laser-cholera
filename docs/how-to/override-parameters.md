# Override parameters

This guide shows how to override individual parameters without forking the bundled `default_parameters.json`. It covers the CLI (`--over key:value`, repeatable), the Python `mods=` dict accepted by [`get_parameters`][laser.cholera.metapop.params.get_parameters], the [`sim_duration`][laser.cholera.utils.sim_duration] helper for shortening the run window, and the visualisation / output flags exposed by the `metapop` entry point.

## Prerequisites

- `laser.cholera` installed in your active environment (`pip install laser.cholera`).
- You have already run the [first-run tutorial](../tutorials/first-run.md) and the `metapop` CLI works in your shell.
- You know which parameter you want to change. The full catalog lives in the [parameter reference](../reference/parameters/index.md).

## Steps

1. Override a single parameter from the CLI by passing one `--over key:value` flag. String values are coerced to the field's declared type by [`override_helper`][laser.cholera.metapop.utils.override_helper] (`int` / `float` / `datetime` / boolean / pass-through for vectors and matrices). Booleans accept any of `true 1 yes y t on enabled` (case-insensitive).

    ```bash
    metapop --seed 20240930 --over phi_1:0.65
    ```

2. Stack multiple overrides by repeating `--over` in the same command. Each `key:value` pair is applied in order on top of the JSON defaults.

    ```bash
    metapop --seed 20240930 \
        --over date_start:2024-01-01 \
        --over date_stop:2024-01-31 \
        --over phi_1:0.65
    ```

3. From Python, pass a `mods` dict to [`get_parameters`][laser.cholera.metapop.params.get_parameters]. The dict layers on top of the bundled defaults (or whatever JSON file you point `get_parameters` at) before validation.

    ```python
    from laser.cholera.metapop.params import get_parameters
    from laser.cholera.metapop.model import Model

    params = get_parameters(mods={"seed": 20240930, "phi_1": 0.65})
    model = Model(params)
    ```

4. For a one-off short run, build a `mods` dict with [`sim_duration`][laser.cholera.utils.sim_duration]. It returns the `date_start` / `date_stop` pair (and any derived window fields) so you don't have to remember which keys to override.

    ```python
    from datetime import datetime
    from laser.cholera.utils import sim_duration
    from laser.cholera.metapop.params import get_parameters

    short_run = sim_duration(datetime(2024, 1, 1), datetime(2024, 1, 31))
    params = get_parameters(mods={"seed": 20240930, **short_run})
    ```

5. Control visualisation and output via the `metapop` flags. These are CLI-only — they configure the entry point, not parameters on the model:

    - `--viz` — display matplotlib visualisations interactively at the end of the run.
    - `--pdf` — write the same visualisations to a PDF in `--outdir`.
    - `--outdir <path>` — directory for any HDF5 results and PDF outputs (defaults to the current directory).
    - `-q` / `--quiet` — suppress the per-tick progress bar.
    - `--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}` — verbosity of the package logger (defaults to `WARNING`).

    ```bash
    metapop --seed 20240930 --viz --pdf --outdir tmp/run-20240930 --loglevel INFO
    ```

!!! warning "Known gotcha — `mods` ndarray coercion"
    `get_parameters(mods={...})` applies overrides via the underlying `PropertySet.__ilshift__` (`<<=`) operator, which bypasses the ndarray coercion that `dict_to_propertysetex` performs on the JSON-load path. As a result, list values for array-typed fields (for example, `S_j_initial: [99990]`) survive untyped into `validate_parameters` and crash with `AttributeError: 'list' object has no attribute 'shape'`.

    Workaround: wrap array-typed values in `np.array(..., dtype=...)` when building the `mods` dict, e.g. `"S_j_initial": np.array([99990], dtype=np.int32)`. A future release is expected to align the two paths so the JSON-load and `mods=` paths produce identically-typed results.

## Full example

```python
from laser.cholera.metapop.model import run_model

# Two scalar overrides applied at load time; `run_model` forwards `**kwargs`
# to `get_parameters` as `mods`. For an array-typed field (e.g. `S_j_initial`)
# wrap the value with `np.array(..., dtype=...)` per the warning above.
model = run_model(None, seed=20240930, phi_1=0.65)
```

And the equivalent CLI invocation, with visualisations written to a PDF:

```bash
metapop --seed 20240930 \
    --over date_start:2024-01-01 \
    --over date_stop:2024-01-31 \
    --over phi_1:0.65 \
    --pdf --outdir tmp/run-20240930 --loglevel INFO
```

## See also

- [First run](../tutorials/first-run.md) — the baseline `metapop` invocation this guide builds on.
- [Single-location tutorial](../tutorials/single-location.md) — end-to-end walkthrough of a minimal configuration.
- [Parameter reference](../reference/parameters/index.md) — every parameter, its type, and its inert value.
- [`override_helper`][laser.cholera.metapop.utils.override_helper] — the string-to-typed-value coercion used by `--over`.
- [`get_parameters`][laser.cholera.metapop.params.get_parameters] — the Python entry point that accepts `mods`.
