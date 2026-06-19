# Calibrate and score

This guide enables the model log-likelihood scoring against observed cases and deaths. It is for the model-user persona working on calibration: you already have a working parameter set, a metapop run that completes, and a stream of observed weekly case / death counts you want to score the model against.

!!! warning "Observed data required"
    Any meaningful likelihood calculation requires observed `reported_cases` and `reported_deaths` arrays of shape `(npatches, n_obs_timesteps)`. The scorer has nothing to score against without them — `calc_likelihood = True` with empty observation arrays produces an undefined result. If you do not have observed data, set `calc_likelihood = False` (the default) and skip this page.

## Prerequisites

- `laser.cholera` installed and a working parameter set (JSON file or in-memory dict) that already produces a successful `run_model` invocation.
- Observed weekly case and death counts for the same patches and date window as the simulation.
- Familiarity with the [reporting parameter reference page](../reference/parameters/reporting.md).

## Steps

1. Prepare two NumPy arrays `reported_cases` and `reported_deaths`, each of shape `(npatches, n_obs_timesteps)`. Missing weeks are supported — non-numeric cells (`None`, `"NA"`, `""`, etc.) are coerced to `np.nan` by [`handle_nan`][laser.cholera.metapop.params.handle_nan] during parameter ingestion, which is why the dtype is `float32` rather than `int`. Patch ordering must match `location_name`.

    ```python
    import numpy as np

    reported_cases = np.array([[5, 8, 12, 7], [3, 6, np.nan, 4]], dtype="float32")
    reported_deaths = np.array([[0, 1, 1, 0], [0, 0, 1, 0]], dtype="float32")
    ```

2. Set `calc_likelihood = True` in the parameter set. This is the single switch the analyzer consults; it is `False` by default.

    ```python
    params_dict["calc_likelihood"] = True
    params_dict["reported_cases"] = reported_cases.tolist()
    params_dict["reported_deaths"] = reported_deaths.tolist()
    ```

3. (Optional) Enable any of the four shape terms by setting its weight to a positive value. All four default to `0` (Negative Binomial core only); `0.25` contributes roughly 25 % as much as the NB core because the terms are T-normalised internally.

    - `weight_peak_timing` — Normal prior on the per-location peak-timing offset in weeks.
    - `weight_peak_magnitude` — log-Normal with adaptive sigma on the observed-vs-estimated peak ratio.
    - `weight_cumulative_total` — NB log-likelihood on cumulative sums at fractional timepoints (25 %, 50 %, 75 %, 100 % of the series by default).
    - `weight_wis` — negated Weighted Interval Score over a set of quantile levels.

    ```python
    params_dict["weight_peak_timing"] = 0.25
    params_dict["weight_cumulative_total"] = 0.25
    ```

4. (Optional) Provide `epidemic_peaks` as a list of records with `iso_code` and `peak_date` columns. This is required by `weight_peak_timing` and `weight_peak_magnitude` and ignored by the other two terms. The list is promoted to a pandas DataFrame by [`dict_to_propertysetex`][laser.cholera.metapop.params.dict_to_propertysetex], which also augments it with a `loc_idx` column (0-based index into `location_name`).

    ```python
    params_dict["epidemic_peaks"] = [
        {"iso_code": "MOZ", "peak_date": "2024-03-15"},
        {"iso_code": "MWI", "peak_date": "2024-04-02"},
    ]
    ```

5. Run the model and read `model.log_likelihood`. The [`Analyzer`][laser.cholera.metapop.analyzer.Analyzer] invokes the scorer on the final tick and writes the scalar onto the model object.

    ```python
    from laser.cholera.metapop.model import run_model

    model = run_model(params_dict)
    print(model.log_likelihood)
    ```

## Scoring a model that has already finished

To compute the likelihood against a model object that was run without `calc_likelihood`, or to re-score with different shape weights, call [`calc_model_likelihood`][laser.cholera.calc_model_likelihood.calc_model_likelihood] directly with the canonical four-array signature:

```python
from laser.cholera.calc_model_likelihood import calc_model_likelihood

nreports = min(
    model.params.reported_cases.shape[1],
    model.patches.incidence.shape[0] - 1,
)
ll = calc_model_likelihood(
    obs_cases=model.params.reported_cases[:, :nreports],
    est_cases=model.results.reported_cases[:, :nreports],
    obs_deaths=model.params.reported_deaths[:, :nreports],
    est_deaths=model.results.reported_deaths[:, :nreports],
    epidemic_peaks=model.params.epidemic_peaks,
    date_start=model.params.date_start,
    date_stop=model.params.date_stop,
    weight_peak_timing=0.25,
)
```

## Full example

```python
import numpy as np
from laser.cholera.metapop.model import run_model
from laser.cholera.metapop.params import get_parameters

params = get_parameters().to_dict()

params["calc_likelihood"] = True
params["reported_cases"] = np.array(
    [[5, 8, 12, 7], [3, 6, np.nan, 4]], dtype="float32"
).tolist()
params["reported_deaths"] = np.array(
    [[0, 1, 1, 0], [0, 0, 1, 0]], dtype="float32"
).tolist()
params["weight_peak_timing"] = 0.25
params["weight_cumulative_total"] = 0.25
params["epidemic_peaks"] = [
    {"iso_code": "MOZ", "peak_date": "2024-03-15"},
    {"iso_code": "MWI", "peak_date": "2024-04-02"},
]

model = run_model(params)
print(model.log_likelihood)
```

## See also

- [Reporting parameters reference](../reference/parameters/reporting.md) — `reported_cases`, `reported_deaths`, `rho`, `rho_deaths`, and the reporting-lag parameters.
- [`calc_model_likelihood`][laser.cholera.calc_model_likelihood.calc_model_likelihood] — full argument list and per-location assembly formula.
- [`Analyzer`][laser.cholera.metapop.analyzer.Analyzer] — the component that wires the scorer into the simulation's final tick.
- [Tutorial: multi-location country run](../tutorials/multi-location-country.md) — note this tutorial deliberately does *not* enable likelihood scoring because the example ships without observed `reported_cases` / `reported_deaths` data.
