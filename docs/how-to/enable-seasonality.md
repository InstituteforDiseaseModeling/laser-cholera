# Enable seasonality on human-to-human transmission

Turn on the two-mode seasonal envelope that modulates the human-to-human transmission rate `beta_j0_hum`. This guide assumes you already have a working `laser-cholera` configuration with non-zero `beta_j0_hum` and just want to add per-patch seasonal forcing.

At each tick `t` and patch `j` the simulator multiplies `beta_j0_hum[j]` by the harmonic

```
1 + a_1_j[j] * cos(2*pi*t/p) + b_1_j[j] * sin(2*pi*t/p)
  + a_2_j[j] * cos(4*pi*t/p) + b_2_j[j] * sin(4*pi*t/p)
```

Both `a_*` and `b_*` are cosine/sine amplitudes — none of them appears in a denominator. The only divide-by-zero hazard is the period `p`, so leave `p` strictly positive. Zeroing all four amplitudes collapses the bracket to a constant `1.0`, which is the canonical "seasonality off" form.

## Prerequisites

- `laser-cholera` is installed and importable as `laser.cholera`.
- You have a parameter dictionary or `mods` overlay that already builds a valid `PropertySet` via `get_parameters(mods=...)`.
- You know `npatches` for your configuration.
- `beta_j0_hum` is non-zero in at least one patch (otherwise the seasonal envelope multiplies zero and has no observable effect).

## Steps

1. Pick the period `p` in days. Use `365` for an annual cycle. `p` must be a positive integer; never set `p = 0` (it is the denominator of `2*pi*t/p`).

    ```python
    mods["p"] = 365
    ```

2. Set the first-mode amplitudes `a_1_j` and `b_1_j`. These give one annual cycle, with `a_1_j` on the cosine and `b_1_j` on the sine. Both are length-`npatches` vectors; typical magnitudes are in the `0.1` to `0.3` range.

    ```python
    mods["a_1_j"] = [0.3] * npatches
    mods["b_1_j"] = [0.1] * npatches
    ```

3. (Optional) Set the second-mode amplitudes `a_2_j` and `b_2_j` to add a semi-annual cycle. Same shape as the first mode; typical magnitudes `0.05` to `0.15`.

    ```python
    mods["a_2_j"] = [0.10] * npatches
    mods["b_2_j"] = [0.05] * npatches
    ```

4. Verify the resulting envelope by evaluating the multiplier across one full period for one patch and plotting it.

    ```python
    import numpy as np
    t = np.arange(365)
    j = 0
    s = (1
         + params.a_1_j[j] * np.cos(2 * np.pi * t / params.p)
         + params.b_1_j[j] * np.sin(2 * np.pi * t / params.p)
         + params.a_2_j[j] * np.cos(4 * np.pi * t / params.p)
         + params.b_2_j[j] * np.sin(4 * np.pi * t / params.p))
    ```

## Turning seasonality back off

Zero all four amplitudes; leave `p` at any positive integer (`365` is fine):

```python
mods["a_1_j"] = np.zeros(npatches)
mods["a_2_j"] = np.zeros(npatches)
mods["b_1_j"] = np.zeros(npatches)
mods["b_2_j"] = np.zeros(npatches)
mods["p"] = 365
```

The harmonic bracket collapses to `1.0` and the per-tick transmission rate becomes a flat `beta_j0_hum`.

## Full example

```python
import numpy as np
from laser.cholera.metapop.params import get_parameters

# Start from the bundled defaults to discover npatches.
import numpy as np

defaults = get_parameters(None, do_validation=False)
npatches = len(defaults.location_name)

mods = {
    "p": 365,
    "a_1_j": np.full(npatches, 0.30, dtype=np.float32),
    "b_1_j": np.full(npatches, 0.10, dtype=np.float32),
    "a_2_j": np.full(npatches, 0.10, dtype=np.float32),
    "b_2_j": np.full(npatches, 0.05, dtype=np.float32),
}
params = get_parameters(mods=mods)

t = np.arange(365)
j = 0
seasonal = (1
            + params.a_1_j[j] * np.cos(2 * np.pi * t / params.p)
            + params.b_1_j[j] * np.sin(2 * np.pi * t / params.p)
            + params.a_2_j[j] * np.cos(4 * np.pi * t / params.p)
            + params.b_2_j[j] * np.sin(4 * np.pi * t / params.p))
```

## See also

- [Human-to-human transmission parameter reference](../reference/parameters/human-transmission.md) — full per-parameter shapes, ranges, and off-values for `beta_j0_hum`, `a_1_j`, `a_2_j`, `b_1_j`, `b_2_j`, and `p`.
- [Seasonality explanation](../explanation/seasonality.md) — why the two-mode harmonic, what each amplitude does to the envelope shape.
- [Single-location tutorial](../tutorials/single-location.md) — Step 5a turns seasonality on inside a guided build.
