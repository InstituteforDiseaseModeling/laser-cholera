# Transmission

This page explains how new cholera infections are generated each day in
`laser.cholera`. The model carries two parallel transmission pathways —
direct human-to-human contact and an environmental (water-reservoir)
pathway — and combines them additively into a single per-patch force of
infection that drives the susceptible-to-exposed transition. The page
assumes a working familiarity with compartmental SIR-style models and
with the parameter groups documented under
[Reference / Parameters](../reference/parameters/index.md); it does not
walk through configuration steps (see the how-to pages for that). After
reading you should understand which terms in the implementation
correspond to which biological story, why both pathways are needed for
cholera in particular, and which knobs in the parameter dictionary
control each pathway's contribution.

## The force of infection and the S to E draw

At every tick, each patch \(j\) accumulates two force-of-infection
contributions — one from infectious humans (`Lambda` in the code) and
one from the environmental reservoir (`Psi`). A susceptible individual
in patch \(j\) is exposed during a tick with probability

$$
P(\text{S} \to \text{E})_{j,t} \;=\; 1 - \exp\!\bigl(-(\Lambda_{j,t} + \Psi_{j,t})\bigr)
$$

The two pathways are sampled separately inside the model — first
`HumanToHuman` draws new infections against `1 - exp(-Lambda)`, then
`EnvToHuman` draws against `1 - exp(-Psi)` using the *already-decremented*
susceptible pool. Drawing them in two passes is numerically equivalent
to drawing once against the combined hazard, and it gives the model
clean separate counts for `incidence_human` and `incidence_env` (both
also summed into a combined `incidence` recorder).

The pipeline order matters. `Susceptible` runs first and writes
`S[tick + 1]` (carrying forward, then applying non-disease deaths and
births). `HumanToHuman` and `EnvToHuman` both *consume* `S[tick + 1]`
rather than `S[tick]`, so each draws against the susceptible pool that
remains after natural mortality (and, for `EnvToHuman`, after the human
pathway has already taken its share). `Environmental` runs last and
advances the reservoir `W[tick + 1]` based on the day's infectious
counts; that updated reservoir is what next tick's `EnvToHuman` will see.

## The human-to-human pathway

The direct pathway treats transmission as a contact process between
infectious and susceptible people, structured by patch and modulated by
seasonality and mobility. The per-tick, per-patch force of infection is

$$
\Lambda_{j,t} \;=\; \frac{\beta^{hum}_{jt} \cdot \bigl(I^{eff}_{j,t}\bigr)^{\alpha_1}}{N_{j,t}^{\alpha_2}}
$$

where \(I^{eff}_{j,t}\) is the *effective* infectious count seen by
patch \(j\), built from both local and visiting infectious individuals:

$$
I^{eff}_{j,t} \;=\; (1 - \tau_j)\,(I^{sym}_{j,t} + I^{asym}_{j,t}) \;+\; \sum_{i \neq j} \pi_{ij}\,\tau_i\,(I^{sym}_{i,t} + I^{asym}_{i,t})
$$

Several pieces of physical intuition are baked into this formula.

**Symptomatic and asymptomatic cases shed equally for direct
transmission.** The model sums `Isym + Iasym` with no weighting; the
asymmetry between the two cohorts shows up only in the environmental
pathway (different shedding rates `zeta_1` vs. `zeta_2`).

**Mobility couples patches.** The fraction `tau_i` of patch \(i\)'s
infectious population is treated as "mobile" and is redistributed across
destinations according to the gravity-model coupling matrix `pi_ij`
(built once at construction time from latitudes, longitudes,
`mobility_omega`, and `mobility_gamma`; see
[mobility](mobility.md)). The remaining `1 - tau_i` stays put. The
implementation in `humantohuman.py:144-148` uses the slightly opaque
`(vec * pi_ij.T).T.sum(axis=0)` idiom to keep the matrix indexed
`[source, destination]`; the result is that an outbreak in a hub patch
spills into well-connected neighbours even before any of its own
residents become infected.

**Seasonality multiplies the baseline.** `beta_jt_human` is a static
`(nticks, npatches)` matrix built once at construction by
`get_daily_seasonality` — it is the product of the per-patch baseline
`beta_j0_hum` and a Fourier bracket
\(1 + a_{1,j}\cos(2\pi t/p) + b_{1,j}\sin(2\pi t/p) + a_{2,j}\cos(4\pi t/p) + b_{2,j}\sin(4\pi t/p)\).
Zeroing the four amplitudes leaves a flat envelope at the baseline;
zeroing `beta_j0_hum` itself disables the entire pathway. See
[seasonality](seasonality.md) for the harmonic in detail.

**The mixing exponents shape how prevalence translates to risk.** With
`alpha_1 = alpha_2 = 1` the formula is *frequency-dependent* — the
relevant quantity is the infectious fraction \(I/N\), independent of
how big the patch is. With `alpha_2 = 0` it is *density-dependent* —
the absolute count of infecteds drives risk. Intermediate values
interpolate. The published MOSAIC parameterisations use `alpha_1` close
to (but slightly below) 1 with `alpha_2 = 1`, i.e. a frequency-dependent
contact process with a mild sub-linear response to prevalence.

Once `Lambda` is assembled it is clipped at zero (a defensive guard
against numerical noise in the seasonal multiplier) and used to draw
new infections as `Binomial(local, 1 - exp(-Lambda))`, where `local` is
the rounded count of *non-emigrating* susceptibles `(1 - tau_i) * S`.
The result is added to `E[tick + 1]`, subtracted from `S[tick + 1]`,
and banked into `incidence_human`.

## The environmental pathway

Cholera also spreads through contaminated water, and `laser.cholera`
models that with a per-patch reservoir \(W_j\) that accumulates from
infectious shedding and decays at a suitability-dependent rate. The
environmental force of infection is

$$
\Psi_{j,t} \;=\; \beta^{env}_{jt} \cdot (1 - \theta_j) \cdot \frac{W_{j,t}}{\kappa + W_{j,t}}
$$

Three terms carry the biology.

**WASH coverage gates exposure.** `theta_j` is the fraction of the
population in patch \(j\) with effective water, sanitation, and hygiene
infrastructure; `(1 - theta_j)` is the fraction that can still acquire
infection from contaminated water. The same factor appears on the
*shedding* side in `environmental.py` (`(1 - theta_j) * shedding`), so
improving WASH simultaneously reduces both contamination and exposure.

**The reservoir saturates.** The term \(W / (\kappa + W)\) is a
Michaelis-Menten-style saturation: when \(W \ll \kappa\), exposure is
roughly linear in \(W\); when \(W \gg \kappa\), it asymptotes at 1
regardless of how much contamination accumulates. `kappa` is the
half-saturation constant in the same units as `W`.

**Suitability modulates the per-tick rate.** The static matrix
`beta_jt_env` is built once at construction time from the baseline
`beta_j0_env` and a per-patch suitability time series `psi_jt`:

```python
psi_bar = psi.mean(axis=0, keepdims=True)
beta_jt_env = beta_j0_env.T * (1.0 + (psi - psi_bar) / psi_bar)
```

The deviation `(psi - psi_bar) / psi_bar` is *centred on zero*, so on
average `beta_jt_env` equals `beta_j0_env`; days more suitable than the
patch's annual mean amplify environmental transmission, less suitable
days dampen it. **Important numerical caveat**: because the formula
divides by `psi_bar`, setting `psi_jt = np.zeros(...)` produces NaN
even when `beta_j0_env = 0`. The canonical "inert" value for
suitability is `psi_jt = np.ones(...)`, which gives `psi_bar = 1` and
collapses the bracket to `1.0`. To switch the environmental pathway off,
zero `beta_j0_env` *and* set `psi_jt` to ones; see
[§4 of the parameter reference](../reference/parameters/environmental-transmission.md)
for the full off-form recipe.

### The reservoir's life cycle

The reservoir itself is owned by `Environmental`, which runs at the
end of each tick (after both transmission components have drawn).
Three things happen per patch:

1. **Carry forward**: `W[tick + 1] := W[tick]`.
2. **Decay**: `W -= Poisson(delta_jt[tick] * W)`, clamped not to exceed
    `W` so the reservoir cannot go negative even on extreme draws. The
    decay rate `delta_jt` is a static matrix derived from `psi_jt` via a
    Beta-CDF interpolator (`map_suitability_to_decay`) between
    `1 / decay_days_short` (high decay, used when suitability is low —
    bacteria die quickly in hostile water) and `1 / decay_days_long`
    (low decay, used when suitability is high — bacteria persist).
    `decay_shape_1` and `decay_shape_2` shape the Beta CDF that smoothly
    interpolates between these extremes.
3. **Shedding**: symptomatic cases contribute
    `(1 - theta_j) * Poisson(zeta_1 * Isym)` and asymptomatic cases
    contribute `(1 - theta_j) * Poisson(zeta_2 * Iasym)`. WASH
    coverage attenuates the deposit, mirroring the gating on the
    exposure side. `zeta_1 > zeta_2` in the published parameterisations,
    so a symptomatic case is the more potent contaminator — by a large
    multiple — but asymptomatics matter because they outnumber
    symptomatics roughly 3-to-1 under the default `sigma`.

The reservoir is thus continuously *recharged* by current infectious
cases and *drained* by decay; under endemic equilibrium these balance
and \(W\) hovers near a steady state. A pulse of incidence drives \(W\)
up sharply; once cases subside, decay (faster in unsuitable conditions)
returns the reservoir toward baseline over days-to-weeks.

## How the two pathways compose

The two pathways are *additive* in the combined hazard
\(\Lambda + \Psi\), but because they are drawn in two consecutive
binomial passes against the same susceptible pool, their effective
contributions interact slightly through the depletion of \(S\) within a
single tick. In practice the per-tick hazards are small enough that
this is a second-order effect, and the implementation choice keeps the
two incidence streams cleanly separable for diagnostics.

Under typical MOSAIC endemic-setting parameterisations the
environmental pathway is the *smaller* of the two when measured by
contribution to incidence — the human pathway dominates day-to-day
transmission, while the environmental pathway provides a slower,
spatially-anchored reservoir of risk that keeps cholera from going
locally extinct between outbreak peaks. In epidemic regimes (or in
patches with very low WASH coverage and high suitability) the balance
can flip; the `incidence_human` vs. `incidence_env` recorders make
this decomposition easy to inspect post-hoc.

### Turning each pathway off

Both pathways have explicit off-forms that the validator accepts and
the simulation honours:

- **Human pathway off**: `beta_j0_hum = np.zeros(npatches)`. The
    seasonal envelope `beta_jt_human` is built as
    `beta_j0_hum * (1 + harmonic)`, so a zero baseline yields a
    permanently zero `Lambda`. The four seasonal amplitudes
    (`a_1_j`, `b_1_j`, `a_2_j`, `b_2_j`) can stay non-zero; they
    multiply zero and contribute nothing.
- **Environmental pathway off**: `beta_j0_env = np.zeros(npatches)`
    *and* `psi_jt = np.ones((nticks, npatches))`. The `ones` value
    avoids the `psi_bar` division-by-zero noted above; the zero
    baseline kills `beta_jt_env` and therefore `Psi`. The reservoir
    `W` will still tick over (shedding minus decay), but no susceptible
    will ever be exposed via it.
- **Both pathways off**: combine the two. The model still runs — births,
    deaths, vaccinations, vital dynamics all continue — but no
    susceptible ever transitions to exposed, so the disease compartments
    drain monotonically to zero as initial infectious cases recover.

These off-forms are useful for unit tests, for isolating one pathway's
contribution in calibration, and for sanity-checking that a scenario
configuration has actually wired up the pathway the user intended.

## How it connects to the rest of the model

The per-tick pipeline (`Susceptible` first, then the other
compartments, then `Census`, then `HumanToHuman`, `EnvToHuman`,
`Environmental`, then `DerivedValues` and `Analyzer`) ensures three
invariants for the transmission step:

- `S[tick + 1]` is fully initialised (with mortality and births
    applied) before either transmission component runs.
- `N[tick]` — the total patch population used in the denominator of
    `Lambda` — is the value `Census` computed for this tick, so
    \(\alpha_2\)-scaling sees the current population, not yesterday's.
- `W[tick]` — used by `EnvToHuman` — is yesterday's reservoir state,
    because `Environmental` writes `W[tick + 1]` only *after*
    `EnvToHuman` has finished. This one-tick lag between shedding and
    exposure is intentional and matches the published MOSAIC formulation.

The Reference parameters that control transmission are grouped under
[human-to-human transmission](../reference/parameters/human-transmission.md)
(`beta_j0_hum`, the four seasonal amplitudes, `p`, `alpha_1`,
`alpha_2`) and
[environmental transmission](../reference/parameters/environmental-transmission.md)
(`beta_j0_env`, `psi_jt`, `theta_j`, `kappa`, `zeta_1`, `zeta_2`,
`decay_days_short`, `decay_days_long`, `decay_shape_1`,
`decay_shape_2`). The mobility-coupling matrix `pi_ij` is built from
[geography-and-mobility](../reference/parameters/geography-and-mobility.md)
parameters (`latitude`, `longitude`, `tau_i`, `mobility_omega`,
`mobility_gamma`).

## See also

- [Human-to-human transmission parameters](../reference/parameters/human-transmission.md)
    — exhaustive parameter reference, including shapes, ranges, and off-values.
- [Environmental transmission parameters](../reference/parameters/environmental-transmission.md)
    — including the `psi_jt = ones` inert-value caveat.
- [Geography and mobility parameters](../reference/parameters/geography-and-mobility.md)
    — `tau_i`, `pi_ij` inputs.
- [Seasonality](seasonality.md) — the Fourier harmonic that builds
    `beta_jt_human`, and the suitability time series `psi_jt` that
    modulates `beta_jt_env` and the environmental decay rate.
- [Mobility](mobility.md) — the gravity model behind `pi_ij` and how
    `tau_i` distributes infectious effort across patches.
- [Configure mobility](../how-to/configure-mobility.md) — task-oriented
    recipes for setting the mobility parameters.
- [Enable seasonality](../how-to/enable-seasonality.md) — task-oriented
    recipes for the seasonal amplitudes and period.
