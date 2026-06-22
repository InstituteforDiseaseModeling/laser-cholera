# R-vs-Python parity tests for `calc_model_likelihood`

These tests address acceptance criterion #7 of [issue #86](../../misc/issue86.md):

> A direct R-vs-Python cross-check on a shared fixture (same obs/est/weights_time/weights_obs/config) agrees to ~1e-8 on the total LL, weighted and unweighted.

Each test builds an R-side fixture, calls `MOSAIC::calc_model_likelihood` (the canonical R implementation, see [`misc/calc_model_likelihood.R`](../../misc/calc_model_likelihood.R)), calls the Python `laser.cholera` implementation via [`reticulate`](https://rstudio.github.io/reticulate/), and asserts that the two scalar log-likelihoods agree within the documented tolerance.

## Prerequisites

```bash
# R packages
Rscript -e 'install.packages(c("testthat", "reticulate", "remotes"))'
Rscript -e 'remotes::install_github("InstituteforDiseaseModeling/MOSAIC-pkg")'

# Python: laser-cholera installed in the project venv
uv pip install -e .
```

### Minimum MOSAIC version

The Python `calc_model_likelihood` tracks the MOSAIC reference at **v0.45.3**
(commit [`7a265b1d`](https://github.com/InstituteforDiseaseModeling/MOSAIC-pkg/commit/7a265b1d),
see [`misc/calc_model_likelihood.R`](../../misc/calc_model_likelihood.R) for the
checked-in copy). The Python implementation has been updated to match this
version, including the new `weights_obs_cases` / `weights_obs_deaths` arguments.

If your installed MOSAIC is older than `0.45.3` the test suite will **skip
cleanly** with a message naming the required version and an `install_github`
command to update. Skipped tests do NOT fail the run — the shell script exits
`0` so CI doesn't false-alarm on a missing dependency. Re-run after updating
MOSAIC to actually exercise the parity assertions.

The version banner printed at the top of every run states the MOSAIC version
detected and whether `weights_obs_cases` is in its function signature, so
mismatches are easy to spot.

## Running the tests

From the repo root:

```bash
tests/R/run_tests.sh
```

The script prints a banner with the version of R, MOSAIC, reticulate, testthat, and laser-cholera being used, then invokes `testthat::test_dir("tests/R", reporter = "summary", stop_on_failure = TRUE)` so the exit code is `0` on full success and non-zero on any failure (suitable for CI).

You can also run the suite directly with `Rscript` (versions banner comes from `helper-setup.R`):

```bash
Rscript -e 'testthat::test_dir("tests/R", reporter = "progress")'
```

## Coverage

| Category | Tests | Tolerance |
|---|---|---|
| NB core only (`weights_obs_*=NULL`) | 9 (base, single-location, NaN observations, non-uniform `weights_location`/`weights_time`/both, `weight_cases`/`weight_deaths` multipliers, 5×20 fixture, custom `nb_k_min_*`) | abs `1e-8` / rel `1e-6` |
| Per-observation weights (issue #86) | 9 (all-ones / mid / heterogeneous / NaN entries / zero-row / deaths-only / both channels / per-cell × `weights_time` / documented-zero-heavy) | abs `1e-8` / rel `1e-6` |
| Shape terms (cumulative, WIS) | 6 (cumulative only, WIS only, custom quantiles, custom timepoints, both together, kitchen-sink) | abs `1e-5` / rel `1e-6` |

## Tolerance rationale

The R and Python implementations call different special-function backends:

- **NB core**: R's `dnbinom(x, size=k, mu=mu)` and Python's `scipy.stats.nbinom.logpmf(x, n=k, p=k/(k+mu))` are algebraically the same kernel and agree to ~1e-8 on the total LL for realistic fixture sizes. Python's `est`-floor at `1e-10` (to avoid `log(0)`) leaks `~1e-10` per perfect-match-zero-data cell; the absolute tolerance comfortably covers this.
- **Shape terms** (peak / cumulative / WIS): the kernels involve `pnorm` / `dlnorm` / `qnbinom` (R) versus `scipy.stats.norm` / `lognorm` / `nbinom.ppf` (Python), which can disagree by a few ULPs per evaluation. Across long sums and quantile-function evaluations the combined drift can reach `1e-6`–`1e-5` on the total LL; the looser tolerance accommodates that without losing the cross-implementation parity guarantee.

## What is NOT cross-checked here

The peak-timing and peak-magnitude shape terms (`weight_peak_timing` / `weight_peak_magnitude`) are **not** covered by these tests. R's `MOSAIC::calc_model_likelihood` consumes a `config = list(location_name, date_start, date_stop, epidemic_peaks)` argument while the Python `calc_model_likelihood` takes `epidemic_peaks` (a pandas DataFrame with a `loc_idx` column), `date_start`, and `date_stop` as separate arguments. The two interfaces would need a tailored fixture translator to compare apples-to-apples. Existing test coverage:

- Python side: [`tests/test_calc_model_likelihood.py`](../test_calc_model_likelihood.py) covers the peak terms via the `epidemic_peaks` DataFrame path.
- R side: MOSAIC's own `testthat` suite covers the peak terms via the `config` path.

A follow-up PR can add a small R-side translator to call Python via `epidemic_peaks` derived from R's `config$epidemic_peaks` and close the last gap.

## Files

- `helper-setup.R` — testthat helper (loaded automatically before any `test-*.R`). Version banner, reticulate setup, `py_calc_ll(...)` wrapper, `expect_ll_match(...)` assertion, common fixture builders.
- `test-python-parity.R` — the test suite (24 tests across the three categories above).
- `run_tests.sh` — CLI driver that prints the version banner and runs the suite with a summary reporter.
