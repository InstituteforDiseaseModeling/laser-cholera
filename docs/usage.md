# Usage

LASIK ships as both a command-line tool (`metapop`) and an importable Python
package (`laser.cholera`). The rest of this page covers the common workflows:
running the model, overriding parameters, computing the model log-likelihood,
and cross-checking against the upstream R reference.

For a single-line installation reference see the
[installation](installation.md) page. For the full list of arguments to any
individual function or class, follow the links into the
[API reference](reference/index.md).

## Quick start

### From the command line

Once installed, the `metapop` entry point runs the default cholera simulation:

```bash
metapop --seed 20240101 --loglevel INFO
```

Common flags:

- `--params <path>` — load parameters from a JSON file instead of the bundled
  defaults.
- `--seed <int>` — set the PRNG seed (default `20241107`).
- `--viz` / `--pdf` — display matplotlib visualisations or write a PDF.
- `--outdir <path>` — directory for any HDF5 or PDF outputs.
- `--over key:value` — override a single parameter, repeatable
  (see [Overriding parameters](#overriding-parameters) below).
- `--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}` — verbosity.
- `-q` / `--quiet` — suppress the per-tick progress bar.

### From Python

[`run_model`][laser.cholera.metapop.model.run_model] accepts a parameter
source in any of four forms — `None`, a `str` path, a `pathlib.Path`, or a
`dict` — and returns the populated model object:

```python
from laser.cholera.metapop.model import run_model

# Use the bundled defaults
model = run_model(None)

# Or pass an in-memory dict (e.g., loaded from your own pipeline)
model = run_model({"seed": 20240101, "loglevel": "INFO"})
```

After `run_model` returns, the per-tick compartment state is available on
`model.people` (e.g., `model.people.S`) and the per-patch derived quantities
are on `model.patches`. The final-tick model log-likelihood (when enabled via
the `calc_likelihood` parameter) is stored on `model.log_likelihood`.

## Overriding parameters

The bundled defaults live in `src/laser/cholera/metapop/data/default_parameters.json`.
You usually don't want to fork that file — instead, override the specific
fields you care about.

From the CLI, repeat `--over key:value`:

```bash
metapop --over seed:20240601 --over date_start:2024-01-01 --over phi_1:0.65
```

The string values are coerced to the right type by
[`override_helper`][laser.cholera.metapop.utils.override_helper] according to
the field's declared type (`int` / `float` / `datetime` / boolean /
pass-through for vectors and matrices). For boolean flags, any of `true 1 yes
y t on enabled` (case-insensitive) means `True`.

From Python, pass a `mods` dict to
[`get_parameters`][laser.cholera.metapop.params.get_parameters]:

```python
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.model import Model

params = get_parameters(mods={"seed": 20240601, "phi_1": 0.65})
model = Model(params)
# ... build components, run, etc.
```

For one-off short runs the helper
[`sim_duration`][laser.cholera.utils.sim_duration] returns a `mods` dict that
shrinks the simulation window:

```python
from datetime import datetime
from laser.cholera.utils import sim_duration

short_run = sim_duration(datetime(2024, 1, 1), datetime(2024, 1, 31))
params = get_parameters(mods=short_run)
```

## Computing the model log-likelihood

The
[`calc_model_likelihood`][laser.cholera.calc_model_likelihood.calc_model_likelihood]
function scores a model fit against observed cases and deaths. It accepts
four 2-D arrays of shape `(n_locations, n_time_steps)` and returns a single
log-likelihood scalar.

### Minimal core call (no shape terms)

The minimal call uses only the Negative Binomial core; all four shape weights
default to `0` (off):

```python
>>> import numpy as np
>>> from laser.cholera.calc_model_likelihood import calc_model_likelihood
>>> obs_cases = np.array([[5, 8, 12, 7], [3, 6, 9, 4]], dtype=float)
>>> est_cases = np.array([[6, 9, 11, 7], [4, 6, 8, 5]], dtype=float)
>>> obs_deaths = np.zeros_like(obs_cases)
>>> est_deaths = np.zeros_like(est_cases)
>>> ll = calc_model_likelihood(obs_cases, est_cases, obs_deaths, est_deaths)
>>> ll < 0
True

```

A perfect match returns the maximum (least-negative) log-likelihood; any
deviation between observed and estimated values reduces it.

### Enabling the shape terms

Set any of the shape-term weights to a positive value to enable additional
calibration signals on top of the NB core. A weight of `0.25` contributes
roughly 25 % as much as the NB core (the terms are T-normalized internally):

- `weight_peak_timing` — Normal prior on the per-location peak timing offset
  in weeks; requires `epidemic_peaks`, `date_start`, and `date_stop`.
- `weight_peak_magnitude` — log-Normal with adaptive sigma on the
  observed-vs-estimated peak ratio; same requirements as peak timing.
- `weight_cumulative_total` — NB log-likelihood on cumulative sums at
  fractional timepoints (defaults to 25 %, 50 %, 75 %, 100 % of the series).
- `weight_wis` — negated Weighted Interval Score on a set of quantile levels.

The `epidemic_peaks` argument is a pandas DataFrame with columns `iso_code`,
`peak_date`, and `loc_idx` (a 0-based integer index into the rows of the
observation arrays). Construction of this DataFrame is handled automatically
by
[`get_parameters`][laser.cholera.metapop.params.get_parameters]
when the incoming JSON has an `epidemic_peaks` entry.

### Integration with the metapopulation model

In normal use the function is invoked from the analyzer
([`Analyzer`][laser.cholera.metapop.analyzer.Analyzer]) on the final tick of
the simulation. The analyzer reads `model.params.reported_cases` /
`reported_deaths` for the observations, `model.results.reported_cases` /
`model.results.reported_deaths` for the estimates, and any per-shape-term
weights from `model.params`.

To enable it for a normal `run_model` invocation, set `calc_likelihood: true`
(plus any shape-term weights you want) in the parameters; the value will end
up on `model.log_likelihood` after the run.

To compute the likelihood against a finished model directly:

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

See
[`calc_model_likelihood`][laser.cholera.calc_model_likelihood.calc_model_likelihood]
for the full argument list and the per-location assembly formula.

## Cross-checking against the upstream R reference

The R implementations the Python port was translated from live under
`reference/` at the repo root (not packaged into the sdist). They are useful
for sanity-checking the Python behaviour when something looks off.

### One-time setup

Install `testthat` and the upstream `MOSAIC` R package:

```bash
Rscript -e 'install.packages("testthat")'
Rscript -e 'remotes::install_github("InstituteforDiseaseModeling/MOSAIC")'
```

If `MOSAIC` is already on your R library path, you can skip the second
command. Verify with:

```bash
Rscript -e 'library(MOSAIC); packageVersion("MOSAIC")'
```

### Run the R test files

The R tests are free-standing `testthat` files rather than an R-package
`tests/testthat/` layout, so invoke them directly with
`testthat::test_file`:

```bash
Rscript -e 'testthat::test_file("reference/test_calc_model_likelihood.R")'
```

Run from the repo root so the relative path resolves. The output is the
standard testthat summary (one dot per assertion, a tally at the end).

Inside an interactive R session use the `progress` reporter for richer output
and easy iteration:

```r
testthat::test_file("reference/test_calc_model_likelihood.R", reporter = "progress")
```

!!! warning "Caveats"
    - `MOSAIC` must be recent enough to include any upstream commits you want
      to cross-check against (e.g., the in-window peak filter).
    - R's `set.seed(123)` does not produce the same RNG stream as Python's
      `np.random.default_rng(123)`. Compare structural properties (finite,
      ordering, inequality) across the two, not exact numerical values — the
      Python test files already follow that convention.

## Where to look next

- [API reference](reference/index.md) — full API reference (autogenerated
  from docstrings).
- [Installation](installation.md) — supported Python versions and the
  development setup.
- [Changelog](changelog.md) — what changed in each release.
