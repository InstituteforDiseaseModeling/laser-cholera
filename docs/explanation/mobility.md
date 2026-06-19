# Mobility

This page explains how `laser.cholera`'s metapopulation model couples its patches in space — how the row-stochastic connectivity matrix `pi_ij` is built from per-patch coordinates, populations, and two scalar exponents, and how that matrix combines with the per-patch exit probability `tau_i` to mix the symptomatic and asymptomatic infectious pools that drive direct (human-to-human) transmission. It assumes a working familiarity with [the SIR-shaped metapopulation overview](model-overview.md) and with the [direct transmission equation](transmission.md); it does not assume any background in transport modelling. After reading you should be able to look at any `pi_ij` heatmap the model produces and reason about why an outbreak seeded in one patch spreads (or does not spread) into another.

## The gravity-model intuition

The connectivity matrix `pi_ij` answers a single question: given a unit of infectious "mixing weight" leaving origin patch `i`, what fraction lands at destination patch `j`? The model uses a *gravity* answer borrowed from spatial-interaction theory: bigger destinations attract more flow, more distant destinations attract less, with both effects tuned by a single power-law exponent.

Concretely, for each ordered pair of distinct patches `(i, j)`, [`get_pi_from_lat_long`][laser.cholera.metapop.utils.get_pi_from_lat_long] computes the unnormalised weight

$$
x_{ij} \;=\; N_j^{\omega} \cdot d_{ij}^{-\gamma},
$$

then row-normalises so every origin distributes a unit of flow exactly once:

$$
\pi_{ij} \;=\; \frac{x_{ij}}{\sum_{k \neq i} x_{ik}}, \qquad \pi_{ii} \;=\; 0, \qquad \sum_{j} \pi_{ij} \;=\; 1 \text{ for each } i.
$$

The two exponents are the only knobs:

- `mobility_omega` ($\omega$) is the *destination-population* exponent. Larger values concentrate flow on the largest neighbours — a small $\omega$ produces a near-uniform spreading; a large $\omega$ funnels everything into the biggest city in the system.
- `mobility_gamma` ($\gamma$) is the *distance-decay* exponent. Small $\gamma$ makes distance almost irrelevant (flow is spread by population alone); large $\gamma$ produces strongly local mixing where only near neighbours matter.

There is intentionally no origin-population term and no overall scale constant: the row normalisation absorbs both. The total *amount* of mixing leaving each patch is controlled separately by `tau_i` (see [Mobility scaling via `tau_i`](#mobility-scaling-via-tau_i) below), and the gravity matrix only sets the *direction* of that flow.

This is the same mathematical family as the `gravity()` migration model shipped in [`laser.core.migration`](https://github.com/InstituteforDiseaseModeling/laser); the upstream `network_{i,j} = k \cdot p_i^a \cdot p_j^b / d_{ij}^c` formula reduces to the `laser.cholera` form when the origin-population exponent is zero (so `p_i^a` drops out), the destination-population exponent matches `omega`, and the distance exponent matches `gamma`, with the overall scale `k` absorbed by the row-normalisation that `laser.cholera` performs and `laser.core.gravity()` does not. `laser.cholera` does **not** delegate to `laser.core.gravity()`; it keeps its own copy in `utils.py`, partly so the row-normalisation and self-flow handling are spelled out, partly because the destination-only weighting is a deliberate simplification of the canonical four-parameter form. The `laser.core` implementation is still a useful sanity check for the formula itself.

## Great-circle distance and the lat/long inputs

The `d_{ij}` term is the great-circle distance between patch centroids, in kilometres, computed by the Haversine formula. `laser.cholera` does *not* re-implement this — it imports `laser.core.migration.distance` (the same module that exposes `gravity()`) and calls it on the full `(latitude, longitude)` vectors of length `npatches`. The returned object depends on the input shape:

- For two or more patches, `distance(...)` returns a symmetric `(npatches, npatches)` matrix with zeros on the diagonal.
- For a single patch (`npatches == 1`), it returns a 0-dimensional scalar — there are no off-diagonal pairs to compute. `get_pi_from_lat_long` detects the scalar case (`if not d.shape:`) and promotes the value to a `(1, 1)` array so downstream broadcasts keep working.

The choice of "centroid" is delegated entirely to the configuration: whatever lat/long pair the configuration JSON specifies for each patch *is* the patch. For real geographies these typically come from a population-weighted centroid of the underlying admin polygon (the [multi-location country tutorial](../tutorials/multi-location-country.md) walks through that derivation for Mozambique's admin-2 districts). The model does not know whether two patches are separated by mountains, oceans, or open border — only by great-circle kilometres.

## Self-coupling and the diagonal

The matrix `pi_ij` describes flow *between* patches; the diagonal is fixed at zero. The implementation handles this in two stages:

1. To compute the off-diagonal weights `x_{ij} = N_j^omega * d_{ij}^(-gamma)` in one broadcast, the code first builds a `d_safe` copy of the distance matrix with `1.0` substituted on the diagonal — that prevents `d^(-gamma)` from producing `inf` where `d = 0`. The full `(npatches, npatches)` weight array is then computed in one shot.
2. Immediately afterwards, the diagonal of `x` is zeroed out: `x[diag] = 0.0`. The placeholder `1.0` on the diagonal of `d_safe` never appears in the final answer; it only kept the intermediate `np.power` call finite.

The row normalisation then divides the (now genuinely off-diagonal) weights by their row sum. This guarantees `pi_ii = 0` and `sum_j pi_ij = 1` for every origin with at least one neighbour.

For the degenerate **single-patch** case the row sum is zero (there are no off-diagonal entries to sum), and the canonical `pi_ij = x_ij / sum_k x_ik` formula is technically undefined. The implementation guards this with the wave-1 performance-rewrite's `np.divide(x, row_sum, where=(row_sum != 0), out=m_hat)`: cells where the row sum is zero are left at the `zeros_like` initial value rather than producing `0/0 = NaN`. The result for a single-patch configuration is the trivial `[[0.0]]` matrix, which is sensible: there is nowhere to migrate to.

## Mobility scaling via `tau_i`

The matrix `pi_ij` is row-stochastic by construction — every row sums to one. It says *where* outflow goes, not *how much* outflow there is. The amount is set, per patch, by the dimensionless exit probability `tau_i`.

`HumanToHuman.__call__` computes the effective infectious pool seen at patch `j` as

$$
I^{\text{eff}}_{j} \;=\; \underbrace{(1 - \tau_j)\,(I^{\text{sym}}_j + I^{\text{asym}}_j)}_{\text{retained at home}} \;+\; \underbrace{\sum_{i \neq j} \tau_i \,\pi_{ij}\,(I^{\text{sym}}_i + I^{\text{asym}}_i)}_{\text{imported from neighbours}}.
$$

In the source this is the three-line block

```python
local_i = (local_frac * total_i).astype(Lambda.dtype)
immigrating_i = ((tau_i * total_i) * pi_ij.T).T.sum(axis=0).astype(Lambda.dtype)
effective_i = local_i + immigrating_i
```

with the `(vector * matrix.T).T` formulation chosen so the result is indexed `[src, dst]` consistently with `pi_ij`. The same exit probability `tau_i` also reduces the susceptible pool at home — only the `local_frac = 1 - tau_i` portion of `S_j` is exposed to the patch's force of infection, mirroring the assumption that a fraction `tau_i` of patch `i` is "elsewhere" on any given tick.

Two limits are worth carrying in your head:

- **`tau_i = 0`** (no mobility): the imported sum vanishes, the retained term is the full infectious pool, and every patch sees only its own infectious population. `pi_ij` is computed but multiplied by zero everywhere — patches behave as fully independent SIR populations.
- **`tau_i = 1`** (full mobility): the retained term vanishes, and every patch sees only its neighbours' infectious populations weighted by `pi_ij`. A patch's own infectious population only re-enters via other patches' incoming weights.

Realistic values are far closer to `0` than to `1` — `tau_i` represents the per-tick probability that someone from patch `i` is contributing to mixing somewhere other than home. The vector is per-patch, so a large capital can be configured with a different exit probability than a remote rural district.

## A validator quirk: `mobility_omega` and `mobility_gamma` are always required

`HumanToHuman.__init__` calls `check_key` on `mobility_omega` and `mobility_gamma` unconditionally, and the `params.py` validator does the same. Neither check is conditioned on whether `tau_i` is the zero vector — the matrix is built every run regardless.

The practical consequence is that even a fully "no-mobility" configuration (every entry of `tau_i` set to zero) still needs `mobility_omega` and `mobility_gamma` set to some valid scalar. The matrix that gets computed in that case is unused (every row of the FOI mixing has `tau_i = 0` and skips the `pi_ij` term), but the validator does not know that. The recommended workaround in inert configurations is to set both exponents to `1.0`, which is a sensible default that costs nothing.

## A single, cached matrix per run

Looking at `HumanToHuman.__init__`:

```python
model.patches.add_array_property("pi_ij", (model.patches.count, model.patches.count), ...)
model.patches.pi_ij[:, :] = get_pi_from_lat_long(model.params)
```

`pi_ij` is built **once**, at component-initialisation time, from the seed initial populations `S_j_initial + E_j_initial + I_j_initial + R_j_initial + V1_j_initial + V2_j_initial`. It is then stored on `model.patches.pi_ij` and reused unchanged for the entire run.

This is a deliberate modelling choice. The gravity weights would in principle drift as births, deaths, and vaccination redistribute the patch totals `N_j`, but in the cholera model the daily vital-rate changes are typically of order $10^{-5}$ per person per tick — too small to meaningfully reshape the destination-attraction term `N_j^omega`. Recomputing every tick would cost an $O(\text{npatches}^2)$ matrix build per call to `HumanToHuman` for no observable change in dynamics. The runtime mobility factor that *does* vary tick-to-tick is the time-dependent `Isym + Iasym` injection into `effective_i`, not the weights `pi_ij` themselves.

If a future use case demanded time-varying connectivity (large vital rates, scheduled travel restrictions, seasonal mobility), the natural place to inject it would be to re-run `get_pi_from_lat_long` on the current `model.patches.N` rather than the seed initial populations, and to do so inside `HumanToHuman.__call__` rather than `__init__`. As shipped, the model assumes the seed totals are good enough for the duration of the run.

## How it connects to the rest of the model

Mobility couples patches in the **human transmission pathway only**:

- `HumanToHuman.__init__` reads `latitude`, `longitude`, `mobility_omega`, `mobility_gamma` and builds `pi_ij` once via `get_pi_from_lat_long`.
- `HumanToHuman.__call__` reads `tau_i` plus the stored `pi_ij` to mix the symptomatic and asymptomatic infectious pools before applying the seasonality envelope `beta_jt_human` and the `alpha_1` / `alpha_2` exponents to produce per-patch `Lambda`.
- New `S -> E` transitions are then drawn from `Binomial(local_frac * S_next, 1 - exp(-Lambda))`, so the retained-at-home `local_frac = 1 - tau_i` also gates which susceptibles are eligible to be infected this tick.

The environmental pathway (`EnvironmentalReservoir` and `EnvToHuman`) is local to each patch — the reservoir state vector `W` evolves independently per patch and is *not* mixed through `pi_ij`. Cross-patch coupling in the model therefore travels exclusively through the infectious-human term; if mobility were turned off (`tau_i = 0` for every patch) the only remaining cross-patch links would be whatever shared seasonality and vital rates the configuration happens to apply.

The relevant Reference parameter group is [Geography and mobility](../reference/parameters/geography-and-mobility.md), which covers the exact shapes and valid ranges for `latitude`, `longitude`, `tau_i`, `mobility_omega`, and `mobility_gamma`. Two adjacent Explanation pages worth reading next are [Direct transmission](transmission.md) (which shows where `pi_ij` and `tau_i` enter the force-of-infection formula) and [Seasonality](seasonality.md) (which covers the multiplicative `beta_jt_human` envelope that gates `Lambda` after the gravity mixing has happened).

## See also

- Reference: [Geography and mobility parameters](../reference/parameters/geography-and-mobility.md) — exact shapes and valid ranges for the five parameters covered here.
- How-to: [Configure mobility](../how-to/configure-mobility.md) — recipe for setting `latitude`, `longitude`, `tau_i`, `mobility_omega`, `mobility_gamma` from a real geography.
- Tutorial: [Multi-location country](../tutorials/multi-location-country.md) — Step 8 visualises a seeded outbreak spreading across Mozambique under the gravity coupling.
- Explanation: [Direct (human-to-human) transmission](transmission.md) — where `pi_ij` and `tau_i` enter the force-of-infection formula.
- Explanation: [Seasonality](seasonality.md) — the `beta_jt_human` envelope that multiplies the gravity-mixed infectious term.
- Upstream reference: [`laser.core.migration`](https://github.com/InstituteforDiseaseModeling/laser) — canonical gravity / Stouffer / radiation implementations; useful for comparing the formula in isolation from the cholera-specific pipeline.
