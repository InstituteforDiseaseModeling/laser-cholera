# Configure mobility

This guide shows you how to set the three parameters that drive inter-patch movement in `laser.cholera`: the per-patch outflow `tau_i`, the destination-population exponent `mobility_omega`, and the distance-decay exponent `mobility_gamma`. It assumes you already have a working multi-patch parameter set and just need to dial mobility in (or turn it off).

## Prerequisites

- `laser-cholera` installed in your environment.
- A working multi-patch configuration — `npatches >= 2`, with per-patch `longitude`, `latitude`, and `N_j_initial` already populated.
- You have completed the [multi-location-country tutorial](../tutorials/multi-location-country.md) or are otherwise comfortable building a `Model` from a parameter dict.

!!! note "`laser.cholera` does not accept a precomputed `pi_ij`"
    The coupling matrix is computed internally each tick from `latitude`, `longitude`, `N`, `mobility_omega`, `mobility_gamma`, and `tau_i` via [`get_pi_from_lat_long`](../reference/parameters/geography-and-mobility.md) in `src/laser/cholera/metapop/utils.py`. There is no parameter slot to inject your own matrix. For a canonical reference implementation of the same gravity kernel, see the gravity model shipped in `laser.core` — useful when verifying the math by hand or when writing tutorials.

## Steps

1. **Turn mobility off (optional).** Set `tau_i` to a zero vector. This is the canonical inert form — patches no longer exchange infectious or environmentally-exposed individuals, and `pi_ij` drops out of the dynamics even though it is still assembled internally.

    ```python
    import numpy as np

    params["tau_i"] = np.zeros(npatches, dtype=np.float32)
    params["mobility_omega"] = 1.0
    params["mobility_gamma"] = 1.0
    ```

    `mobility_omega` and `mobility_gamma` are still required by `validate_parameters` — there is no off-form for these two — so leave them at any sane float (e.g. `1.0`). They have no effect on the run when `tau_i = 0`.

2. **Set per-patch `tau_i`.** Each element is the proportion of the patch leaving per tick, constrained to `[0, 1]`. Typical values are `0.001`–`0.05`. A flat vector is fine for an illustrative configuration:

    ```python
    params["tau_i"] = np.full(npatches, 0.01, dtype=np.float32)
    ```

3. **Choose `mobility_omega`.** This scalar is the exponent applied to the destination population in the gravity kernel. Higher values bias flow toward more populous destinations. Typical range `0.5`–`2.0`:

    ```python
    params["mobility_omega"] = 1.0
    ```

4. **Choose `mobility_gamma`.** This scalar is the exponent applied to the distance penalty. Higher values make flow drop off faster with distance. Typical range `1.0`–`3.0`:

    ```python
    params["mobility_gamma"] = 1.5
    ```

5. **Verify `pi_ij` after the run.** After `model.run()`, inspect `model.patches.pi_ij`. Its shape is `(npatches, npatches)`, the diagonal is zero, and each row sums to 1:

    ```python
    pi_ij = model.patches.pi_ij
    assert pi_ij.shape == (npatches, npatches)
    assert np.allclose(pi_ij.sum(axis=1), 1.0)
    ```

## Full example

```python
import numpy as np

from laser.cholera.metapop import Model

npatches = params["npatches"]

params["tau_i"] = np.full(npatches, 0.01, dtype=np.float32)
params["mobility_omega"] = 1.0
params["mobility_gamma"] = 1.5

model = Model(params)
model.run()

pi_ij = model.patches.pi_ij
assert pi_ij.shape == (npatches, npatches)
assert np.allclose(pi_ij.sum(axis=1), 1.0)
```

## See also

- [Geography & mobility parameter reference](../reference/parameters/geography-and-mobility.md) — full parameter spec, shapes, validator behaviour.
- [Mobility explanation](../explanation/mobility.md) — the gravity-model derivation of `pi_ij` and how `tau_i`, `mobility_omega`, `mobility_gamma` enter the force-of-infection.
- [Multi-location country tutorial](../tutorials/multi-location-country.md) — Step 8 shows mobility spreading a seeded outbreak via `pi_ij`.
