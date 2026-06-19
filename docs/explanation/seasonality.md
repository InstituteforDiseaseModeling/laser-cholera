# Seasonality

This page explains the seasonality envelope that modulates human-to-human cholera transmission in the metapop model: the two-mode Fourier harmonic implemented in [`get_daily_seasonality`][laser.cholera.metapop.utils.get_daily_seasonality], why it has the shape it does, how the four amplitude vectors plus the period parameterise it, and how it plugs into the transmission pipeline. It assumes you have a working mental model of the per-tick metapop loop (see [model-overview.md](model-overview.md)) and at least passing familiarity with Fourier series. By the end you should be able to read any choice of `a_1_j`, `b_1_j`, `a_2_j`, `b_2_j`, `p` and have an intuition for the seasonal pattern it will produce — and the corollary, that you should be able to translate an observed seasonal pattern (single-peak vs two-peak, when the peak falls in the year, how big the swing is) into amplitudes you can drop into a parameters file.

## The harmonic form

`get_daily_seasonality` (in `src/laser/cholera/metapop/utils.py`) computes a `(nticks, npatches)` matrix `beta_jt_human` by evaluating, for each tick `t` and each patch `j`,

$$
\beta^{hum}_{jt} \;=\; \beta^{hum}_{j0} \cdot \Big( 1 \;+\; a_{1,j}\cos\tfrac{2\pi t}{p} \;+\; b_{1,j}\sin\tfrac{2\pi t}{p} \;+\; a_{2,j}\cos\tfrac{4\pi t}{p} \;+\; b_{2,j}\sin\tfrac{4\pi t}{p} \Big)
$$

with `t` 1-indexed to match the upstream R reference implementation. The `1 +` baseline is the load-bearing piece: it means the multiplier hovers around 1, and the four amplitudes drive deviations above and below that baseline. With all four amplitudes zero, the bracket collapses to `1.0` and the seasonal envelope is just a flat `beta_j0_hum_j` for every tick of the run — no seasonality, but still a non-zero transmission rate. That property — that "no seasonality" is a clean special case of "some seasonality" — is the reason the envelope is written this way rather than as an additive offset.

The output is a `float32` matrix that is baked once, at `HumanToHuman.__init__` time, and stored on `model.patches.beta_jt_human`. It does not change during the run; `humantohuman.py` indexes into it by tick. That matters for performance (no per-tick trigonometry) and for predictability (the seasonality profile is fully determined by the parameter set, with no stochastic drift).

## Why two modes

Why two modes — why not one, or four? The answer is empirical. Cholera transmission in most settings follows a primary annual cycle: a single rainy season, a single dry season, one peak in transmission per calendar year. That is what mode 1 (angular frequency `2*pi/p`) captures. But in some settings — South Asia and parts of West Africa are the textbook examples — the year contains *two* transmission peaks, typically a smaller pre-monsoon peak and a larger post-monsoon peak, separated by roughly half a year. Mode 2 (angular frequency `4*pi/p`) is what lets the envelope produce a second peak per cycle.

Two modes are enough to fit either of those patterns cleanly:

- Single-peak years: a non-zero mode-1 amplitude is sufficient. Set mode 2 to zero and the envelope rises and falls once per period.
- Two-peak years: a non-zero mode-2 amplitude alongside mode 1 produces the characteristic double-bump per year. The relative magnitudes of the two modes determine how prominent each peak is.

You could in principle add a mode 3 (frequency `6*pi/p`, three peaks per year), but cholera surveillance data rarely supports anything beyond second-order structure, so the implementation stops at mode 2. Higher modes would also be increasingly hard to identify from noisy weekly case data.

## Per-patch amplitudes

Each of `a_1_j`, `b_1_j`, `a_2_j`, `b_2_j` is a length-`npatches` vector. Every patch gets its own seasonal shape. This is structurally important: cholera seasonality in a country like Mozambique differs by latitude (rainy season timing shifts north-to-south), and the multi-admin configuration is intended to capture that.

In practice, two common patterns show up:

- **Country-wide broadcast**: when patches are admin units within one country and seasonality data is only available at country level, the same four scalars are broadcast across all `npatches` entries. Every patch then has identical seasonal shape, but population, mobility, and environmental terms still produce per-patch trajectories.
- **Per-country variation**: when patches span multiple countries (e.g. a regional cholera analysis), each country's patches get that country's amplitudes; the MOSAIC default-parameters table is structured this way.

There is no constraint that amplitudes be similar across patches — the validator only checks that each vector has length `npatches`. Wildly different per-patch shapes are allowed; whether they are *defensible* is a calibration question, not a model-engine one.

## Period `p`

The period `p` is the cycle length in ticks. Since the model runs in unit-day ticks (one tick per day), `p = 365` gives an annual cycle and is the default. The validator (in `params.py`) checks that `p` is integer-valued but does not constrain its magnitude. In particular `p = 0` would produce a division by zero in the cosine and sine arguments; nothing in the model catches that — it is the user's responsibility to keep `p > 0`. Negative periods are likewise unguarded but mathematically equivalent to flipping the signs on the sine terms, which is rarely what you want.

There is no per-patch period: all patches share the same `p`. If you ever needed per-patch periods (e.g. modelling a region where the seasonal cycle has different lengths in different ecological zones), you would need to extend `get_daily_seasonality` to make `p` a vector.

## Amplitude versus phase

The natural parameterisation of a single harmonic mode is amplitude and phase: how big the swing is, and when in the cycle the peak falls. The natural parameterisation for *fitting* (and for the linear algebra inside the model) is a cosine coefficient and a sine coefficient. These are equivalent:

$$
a \cos\theta + b \sin\theta \;=\; A \cos(\theta - \varphi)
$$

with $A = \sqrt{a^2 + b^2}$ and $\varphi = \mathrm{atan2}(b, a)$. So for mode 1, the effective amplitude is $\sqrt{a_{1,j}^2 + b_{1,j}^2}$ and the phase is $\mathrm{atan2}(b_{1,j}, a_{1,j})$. To go the other way — author who thinks in (amplitude, phase) and wants to back out (a, b) — use:

$$
a = A \cos\varphi, \qquad b = A \sin\varphi.
$$

The phase, expressed as a fraction of the year, picks out where the peak of mode 1 sits. Phase 0 (i.e. `b = 0`, `a > 0`) puts the peak at `t = 0`. Positive phase shifts the peak later in the year.

Authors should be aware that the effective seasonal swing is bounded by the *sum* of the two mode amplitudes: in the worst case, both modes can constructively interfere and the bracket can reach `1 + A_1 + A_2`. If $A_1 + A_2 > 1$, the bracket can also go *negative*, which would imply a negative transmission rate. `humantohuman.py` defends against that by clipping the rate at zero, but a sustained negative bracket suggests miscalibrated amplitudes rather than a feature to rely on.

## Disabling seasonality (and what *not* to worry about)

To switch seasonality off without disabling human-to-human transmission entirely, zero all four amplitude vectors:

```python
mods = {
    "a_1_j": np.zeros(npatches),
    "b_1_j": np.zeros(npatches),
    "a_2_j": np.zeros(npatches),
    "b_2_j": np.zeros(npatches),
}
```

This collapses the bracket to `1.0` for every tick and every patch. `p` stays at its default (`365`); leaving `p` alone is important because it appears in the denominator of the cosine/sine arguments.

There is no division-by-zero hazard in the four amplitudes themselves. Neither `a_*_j` nor `b_*_j` appears in a denominator anywhere in `get_daily_seasonality` — they are pure multiplicative coefficients on cosine and sine terms. (An earlier draft of the documentation plan suggested zeroing `b_*_j` would divide by zero; that was incorrect and has been verified against the code.) The only place a divide-by-zero can hide in this subsystem is `p = 0`, which the validator does not catch.

If you want to disable human-to-human transmission *entirely* — not just the seasonal modulation but the whole pathway — zero `beta_j0_hum` instead. Since the harmonic bracket is multiplied by `beta_j0_hum`, that propagates through and forces every entry of `beta_jt_human` to zero regardless of the amplitudes.

## Multiplicative composition into the force of infection

The seasonality envelope multiplies the baseline transmission rate to form `beta_jt_human`, and that is the only place seasonality enters the model. `HumanToHuman.__call__` (in `humantohuman.py`) reads the current row `beta_jt_human[tick, :]` once per tick and uses it as the `beta` coefficient in the force-of-infection formula:

$$
\Lambda_{j,t+1} \;=\; \frac{\beta^{hum}_{jt}\,\big((S_{jt}(1 - \tau_j))(I_{jt}(1 - \tau_j) + \sum_{i \neq j}\pi_{ij}\,\tau_j\,I_{it})\big)^{\alpha_1}}{N_{jt}^{\alpha_2}}.
$$

Note three things about this composition. First, seasonality is *multiplicative*, not additive: a tick with `m_j(t) = 1.2` produces a force of infection 20% higher than a tick with `m_j(t) = 1.0`, all else equal. Second, seasonality only touches the *human-to-human* pathway. The environmental pathway has its own time-varying multiplier (`psi_jt`, the environmental suitability) and a separate baseline (`beta_j0_env`); seasonality as discussed on this page does not modulate it. Third, because the envelope is baked once at init and then indexed per-tick, the cost of having seasonality on is essentially zero at runtime — a single array lookup per tick per patch.

## How it connects to the rest of the model

In the per-tick pipeline:

1. **At init time**: `HumanToHuman.__init__` calls `get_daily_seasonality(model.params)` once, producing the `(nticks, npatches)` matrix `model.patches.beta_jt_human`. This is the only consumer of `beta_j0_hum`, `a_1_j`, `b_1_j`, `a_2_j`, `b_2_j`, and `p` directly.
2. **Per tick**: `HumanToHuman.__call__` reads `beta_jt_human[tick, :]` and uses it as the per-patch `beta` in the Lambda formula above.
3. **Nowhere else**: the environmental transmission component, the vital dynamics component, and the vaccination component do not touch the seasonality matrix.

The reference page that lists the controlling parameters is [reference/parameters/human-transmission.md](../reference/parameters/human-transmission.md). The how-to that walks through turning seasonality on for a real configuration is [how-to/enable-seasonality.md](../how-to/enable-seasonality.md). The explanation page that covers the rest of the human-to-human transmission term — gravity-based mixing, the `alpha_1` / `alpha_2` non-linearities, the role of `tau_i` — is [transmission.md](transmission.md).

## See also

- [reference/parameters/human-transmission.md](../reference/parameters/human-transmission.md) — the parameter reference for `beta_j0_hum`, `a_1_j`, `b_1_j`, `a_2_j`, `b_2_j`, `p`, `alpha_1`, `alpha_2`.
- [how-to/enable-seasonality.md](../how-to/enable-seasonality.md) — recipe for turning the two-mode harmonic on and resetting the inert baseline.
- [tutorials/single-location.md](../tutorials/single-location.md) — Step 5a turns seasonality on for a single patch and shows the resulting `beta_jt_human` series.
- [explanation/transmission.md](transmission.md) — how `beta_jt_human` enters the force-of-infection formula and combines with mobility, infectious counts, and the `alpha_1` / `alpha_2` exponents.
- [explanation/mobility.md](mobility.md) — the other per-patch modulation on human-to-human Lambda (gravity-derived `pi_ij`, `tau_i`).
