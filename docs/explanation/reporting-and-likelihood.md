# Reporting and likelihood

This page explains how `laser.cholera` turns the simulation's internal compartmental state into the
*observable* time series that calibration and scoring actually compare against, and how those
observables are then folded into a single scalar log-likelihood. It assumes you have read
[Model overview](model-overview.md) and are comfortable with the rough shape of an SEIRV step. By
the end you should be able to answer three questions: which random draws separate "true" cases
from "reported" cases, what the four-term likelihood ensemble actually measures, and where each
parameter on the reporting / scoring surface enters the math.

## Reporting is a thinning of the truth

Every tick, the [`Infectious`][laser.cholera.metapop.infectious.Infectious] component samples a
fresh batch of new symptomatic incidences from the `E -> Isym` progression draw:

$$
\Delta I^{\mathrm{sym}}_{j, t+1} \;=\; \mathrm{round}\!\bigl(\sigma_j \cdot \mathrm{Binom}(E_{j,t+1},\, 1 - e^{-\iota})\bigr).
$$

These are stored on `model.patches.new_symptomatic[t+1]` — they are the *true* incident
symptomatic infections in patch $j$ on tick $t+1$. The reporting transform then converts them into
the observable count `reported_cases` via two stochastic steps:

$$
R^{\mathrm{cases}}_{j, t+1} \;=\; \mathrm{round}\!\left(\frac{\mathrm{Binom}\bigl(\Delta I^{\mathrm{sym}}_{j,\, t - \delta_R^{\mathrm{cases}}},\; \rho\bigr)}{\chi^{\mathrm{eff}}_{j,t}}\right).
$$

There are three knobs here, each doing a distinct job:

- **The detection probability $\rho$.** A scalar in $[0, 1]$. Each truly-symptomatic case has an
  independent $\rho$ chance of being detected. With $\rho = 1$ every symptomatic case is
  detected (modulo the $\chi$ inflation, below); with $\rho = 0.1$ only ~10% are. This is a
  classical observation-process thinning.
- **The reporting lag $\delta_R^{\mathrm{cases}}$.** An integer number of ticks. The binomial is
  drawn from the incidence at tick $t - \delta_R^{\mathrm{cases}}$, not at tick $t$. This is how
  the model represents the delay between case onset and the case appearing in surveillance
  totals.
- **The healthcare-access modifier $\chi^{\mathrm{eff}}_{j,t}$.** A per-patch scalar in $(0, 1]$
  that *divides* the detected count. Because it sits in the denominator, $\chi^{\mathrm{eff}} <
  1$ *inflates* reported cases above the binomial draw, and $\chi^{\mathrm{eff}} = 1$ is the
  identity. The intuition: in an established outbreak, surveillance attention rises and the same
  underlying case gets logged in more streams; $\chi$ is a coarse stand-in for that effect.

Reported deaths follow the identical pattern, with two substitutions: `disease_deaths` in place
of `new_symptomatic`, the scalar `rho_deaths` in place of `rho`, and the integer
`delta_reporting_deaths` in place of `delta_reporting_cases`. Critically, the death stream does
*not* divide by $\chi$ — the implementation at
[`infectious.py:209-212`][laser.cholera.metapop.infectious.Infectious] applies only the binomial
detection step. If you want a $\chi$-style inflation on deaths you would have to bake it into
`rho_deaths`.

### The regime-switching factor $\chi^{\mathrm{eff}}$

The effective $\chi$ is picked per patch, per tick, from a two-way switch:

$$
\chi^{\mathrm{eff}}_{j,t} \;=\; \begin{cases} \chi^{\mathrm{epidemic}} & \text{if } I^{\mathrm{sym}}_{j,\, t-\delta_R^{\mathrm{cases}}} / N_{j,\, t-\delta_R^{\mathrm{cases}}} \;\ge\; \theta^{\mathrm{epi}}_j, \\ \chi^{\mathrm{endemic}} & \text{otherwise.} \end{cases}
$$

The lagged symptomatic prevalence is compared against `epidemic_threshold`; this is the same
quantity used to decide whether the per-patch disease-mortality multiplier kicks in. To make the
reporting transform a no-op, set `chi_endemic == chi_epidemic == 1.0`. (To disable
**regime-switching** generally — which is a separate concept than disabling reporting — set
`chi_endemic == chi_epidemic` *and* `mu_j_epidemic_factor == 0`; see
[Regime switching](../reference/parameters/regime-switching.md) for the full inert configuration.)

### `delta_reporting_cases` does double duty

It is worth being explicit about this because it surprises people. `delta_reporting_cases` is
read twice every tick:

1. At [`infectious.py:196-201`][laser.cholera.metapop.infectious.Infectious], to fetch the
   *lagged* symptomatic prevalence `model.people.Isym[t - delta_reporting_cases]` for the
   epidemic-regime flag that controls the disease-mortality multiplier $\mu_{jt}$.
2. At [`infectious.py:260-268`][laser.cholera.metapop.infectious.Infectious], to fetch the
   *lagged* `new_symptomatic[t - delta_reporting_cases]` that gets thinned by $\rho$ for
   `reported_cases`.

So changing `delta_reporting_cases` shifts both the regime probe and the reporting lag together.
This is intentional: both effects are about "what surveillance saw recently," and modelling them
on separate clocks would create awkward edge cases where a patch is in epidemic mortality regime
but its reports are still pre-outbreak. Be aware that you cannot pull the two apart with
parameters alone.

## Observed vs. estimated: the scoring game

The likelihood machinery is a game between two arrays of the same shape:

- **Observed**: `params.reported_cases` and `params.reported_deaths`, shape
  `(npatches, n_obs_timesteps)`. These come from real surveillance data and are NaN-tolerant —
  missing weeks are dropped from the per-location likelihood, not zero-imputed.
- **Estimated**: `results.reported_cases` and `results.reported_deaths`, written by the
  reporting transform above on each simulated tick.

The [`Analyzer`][laser.cholera.metapop.analyzer.Analyzer] component invokes
[`calc_model_likelihood`][laser.cholera.calc_model_likelihood.calc_model_likelihood] on the final
tick, comparing the first `min(obs.shape[1], estimate.shape[0] - 1)` columns of each. The result
is stashed on `model.log_likelihood`.

A few important consequences flow from this design:

- **Without observed data, the likelihood is undefined.** If `params.reported_cases` is missing
  or `params.calc_likelihood` is `False`, `model.log_likelihood` is set to `nan`. If the
  likelihood call raises `ValueError` (mismatched shapes, negative weights, etc.) it is set to
  `-inf` so calibration loops can detect and skip the failed evaluation without try/except at
  every call site. See the warning admonition in
  [How to: calibrate and score](../how-to/calibrate-and-score.md).
- **The score is computed once.** Per-tick likelihoods are not tracked. If you want a likelihood
  trajectory over time, you have to compute it externally from `results.reported_cases`.
- **Estimated counts are stochastic.** Two runs of the same parameter set with different seeds
  will produce different scores. Calibration that ignores this Monte-Carlo noise — e.g. by
  running each candidate parameter set only once — will be noisy on tight likelihood landscapes.

## The four-term likelihood ensemble

When `params.calc_likelihood = True`, the per-location, per-outcome Negative Binomial
log-likelihood is *always* computed. It is the core of the score. On top of it, four optional
**shape terms** can be enabled by giving them positive weights; each contributes additively. All
default weights are zero, so a fresh parameter set scores on the NB core alone.

### Term 1: the NB core

For each location $j$ and outcome $\{\mathrm{cases}, \mathrm{deaths}\}$, the model evaluates a
weighted Negative Binomial log-PMF:

$$
\mathrm{NB}_{j} \;=\; \sum_{t} w_t \cdot \log p_{\mathrm{NB}}\!\bigl(\mathrm{obs}_{j,t} \;\big|\; \mu = \mathrm{est}_{j,t},\; k\bigr).
$$

The dispersion $k$ is *estimated from the observed data* by weighted method-of-moments
(`nb_size_from_obs_weighted`), with a floor `nb_k_min_cases` / `nb_k_min_deaths` (default `3`)
and a ceiling at $10^5$. Because $k$ depends only on the observations, two competing parameter
sets are scored against the same noise model — fitting "to the noise" is not a degree of
freedom.

When the estimated variance does not exceed the mean (a Poisson or sub-Poisson regime),
`nb_size_from_obs_weighted` returns `inf` and the NB log-PMF collapses to the Poisson log-PMF.
This is the safest fallback: it scores the location, but stops pretending overdispersion exists
where the data does not show any.

### Term 2: peak timing

If `weight_peak_timing > 0`, for each known peak in `epidemic_peaks` whose `peak_date` falls
inside `[date_start, date_stop]`, the model locates the *estimated* peak inside a $\pm 14$-step
window around the observed peak and scores the timing offset (converted to weeks) with a
Normal(0, `sigma_peak_time`) log-PDF. `sigma_peak_time` defaults to 1 week.

This term penalises a model that gets the right magnitude but in the wrong calendar quarter —
something the NB core does only weakly, because a one-bin shift in the peak hits a small handful
of timesteps but does not flatten the whole likelihood profile.

### Term 3: peak magnitude

If `weight_peak_magnitude > 0`, the model takes the log-ratio of the estimated to the observed
peak height inside the same $\pm 14$-step window and scores it with a Normal(0, $\tilde\sigma$)
log-PDF on the log scale. The standard deviation is *adaptive*:

$$
\tilde\sigma \;=\; \sigma^{\mathrm{peak,log}} \cdot \sqrt{\frac{100}{\max(\mathrm{obs\_peak}, 100)}}.
$$

Large observed peaks tighten the prior; small observed peaks (under 100) leave $\tilde\sigma$ at
its base value `sigma_peak_log` (default `0.5`). The intuition: when the observed peak is in the
hundreds or thousands of cases, the relative noise is small and the model should be held to a
tight magnitude match; when the observed peak is a handful of cases, the relative noise is
enormous and the term should not dominate the score.

### Term 4: cumulative progression

If `weight_cumulative_total > 0`, the model evaluates the NB log-PMF of the *cumulative* obs vs.
estimated counts at a sequence of fractional timepoints, by default at 25%, 50%, 75%, and 100%
of the series length. The dispersion is scaled by the number of summed timesteps
($k \cdot \mathrm{end\_idx}$), and the contribution from each timepoint is normalised by
$\mathrm{end\_idx}$ before averaging — this makes the cumulative term roughly per-observation
and comparable in scale to the NB core.

This term penalises models that get the per-tick shape mostly right but accumulate a systematic
under- or over-count over the season.

### Term 5: WIS

If `weight_wis > 0`, the model computes the **Weighted Interval Score** at a set of quantile
levels (default $\{0.025, 0.25, 0.5, 0.75, 0.975\}$). The WIS uses the NB (or Poisson, when $k$
collapses) quantile function evaluated at each timestep to build symmetric prediction intervals,
then scores each observation against those intervals with an interval-coverage penalty plus a
median absolute-error term. Lower WIS is better, so the term is negated before being added to
the score.

### T-normalisation: putting the four terms on a common scale

The shape terms are internally **T-normalised** so that the user-supplied weights live on a
common scale:

$$
\ell_j \;=\; w_c \mathrm{NB}_c + w_d \mathrm{NB}_d \;+\; \frac{N_{\mathrm{obs}}}{N_{\mathrm{peaks}}} \cdot w_{pt}\,(\cdots) + \frac{N_{\mathrm{obs}}}{N_{\mathrm{peaks}}} \cdot w_{pm}\,(\cdots) + \frac{N_{\mathrm{obs}}}{N_{\mathrm{cum}}} \cdot w_{\mathrm{cum}}\,(\cdots) + \frac{N_{\mathrm{obs}}}{N_{\mathrm{q}}} \cdot w_{\mathrm{wis}}\,(\cdots)
$$

The ratio $N_{\mathrm{obs}} / N_{\mathrm{component}}$ scales each shape term so that with a
weight of `1.0`, the term contributes roughly as much signal as the NB core. A weight of `0.25`
therefore means *about a quarter as much signal* as the core. This makes the weights interpretable
without knowing the exact length of the observation series in advance.

The location-level total $\ell_j$ is multiplied by `weights_location[j]` and summed over all
locations to produce the scalar score.

## How `epidemic_peaks` flows through the system

The peak-timing and peak-magnitude terms both need a list of *known* epidemic peaks per
location. That list lives in `params.epidemic_peaks`, a pandas DataFrame with columns `iso_code`
and `peak_date`. Its trip through the system has three stages worth knowing about:

1. **Ingestion.** `dict_to_propertysetex` (the helper that builds `params` from the raw config)
   converts the `epidemic_peaks` field to a DataFrame and auto-adds a `loc_idx` column that maps
   each `iso_code` to the integer row in `params.location_name`. So by the time
   `calc_model_likelihood` sees the DataFrame, every peak knows which simulation row it belongs
   to.
2. **In-window filter.** Inside `calc_model_likelihood`, any peak whose `peak_date` falls outside
   `[date_start, date_stop]` is silently dropped. This matters because the naive
   `np.argmin(|date_seq - peak_date|)` would otherwise snap an out-of-window peak to either
   `t=0` or `t=n_time_steps-1`, biasing the shape terms. This was a post-`f8e3b38` correction
   that brought the Python implementation into line with the upstream MOSAIC R code.
3. **Bypass.** If neither `weight_peak_timing` nor `weight_peak_magnitude` is positive, the
   `epidemic_peaks` DataFrame is never read; the field can be omitted from the parameter set
   entirely. The validator only checks the column schema when the field is *present*.

If `epidemic_peaks` is present but `date_start` / `date_stop` are missing — or if the daily /
weekly date sequence doesn't match the simulation length — `_peak_idx_lists` stays `None` and
both peak terms silently contribute zero. The simulation still runs and the score is still
produced; you just don't get any peak-shape signal.

## How it connects to the rest of the model

The reporting transform lives inside the [`Infectious`][laser.cholera.metapop.infectious.Infectious]
component, which fires once per tick during the main loop. By the end of the run,
`results.reported_cases` and `results.reported_deaths` are fully populated.

On the final tick, the [`Analyzer`][laser.cholera.metapop.analyzer.Analyzer] component fires and
invokes `calc_model_likelihood`, pulling all available optional arguments out of `params` (any of
`weight_cases`, `weight_deaths`, `weights_time`, `weights_location`, `nb_k_min_cases`,
`nb_k_min_deaths`, the four `weight_*` shape weights, `sigma_peak_time`, `sigma_peak_log`,
`epidemic_peaks`, `date_start`, `date_stop`). The scalar result is written to
`model.log_likelihood`. From there it is yours to consume — calibration sweeps typically pickle
the model, read `log_likelihood`, and discard the rest.

The reporting machinery is downstream of the dynamics but tightly coupled to one piece of them:
the regime-switching factor $\chi^{\mathrm{eff}}$ shares the epidemic threshold with the
disease-mortality multiplier $\mu_{jt}$. If you want to reason about either in isolation, see
[Regime switching](../reference/parameters/regime-switching.md) — and remember that the
canonical *inert* values for the lagged regime probe are `chi_endemic == chi_epidemic` *and*
`mu_j_epidemic_factor == 0`, not zero $\chi$ values (which would cause a divide-by-zero in the
reporting transform).

## See also

- [Reporting parameter reference](../reference/parameters/reporting.md) — exhaustive per-field
  list of `rho`, `rho_deaths`, `delta_reporting_cases`, `delta_reporting_deaths`,
  `calc_likelihood`, the weight / sigma scoring fields, and `epidemic_peaks`.
- [Regime switching parameter reference](../reference/parameters/regime-switching.md) —
  `chi_endemic`, `chi_epidemic`, `epidemic_threshold`, `mu_j_epidemic_factor`, and the inert
  configurations for both reporting inflation and epidemic-mortality regimes.
- [How to: calibrate and score](../how-to/calibrate-and-score.md) — operational recipe for
  setting `calc_likelihood = True`, supplying observed data, choosing shape-term weights, and
  reading the scalar score off `model.log_likelihood`.
- [Model overview](model-overview.md) — where the reporting transform sits in the per-tick
  pipeline and how it relates to the SEIRV state.
