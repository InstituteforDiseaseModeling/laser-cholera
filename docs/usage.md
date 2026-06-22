# Usage

LASIK ships as both a command-line tool (`metapop`) and an importable Python
package (`laser.cholera`). This page covers the minimal "just run the
defaults" path. For deeper workflows — overriding parameters, scoring
against observed data, building configurations from scratch — see the
[tutorials](tutorials/first-run.md), the [how-to guides](how-to/override-parameters.md),
or jump directly into the [parameter reference](reference/parameters/index.md).

For installation see [Installation](installation.md). For the full API see
the [API reference](reference/index.md).

## Quick start

### From the command line

Once installed, the `metapop` entry point runs the bundled cholera
simulation:

```bash
metapop --seed 20240930 --loglevel INFO
```

Common flags (see [How-to › Override parameters](how-to/override-parameters.md)
for the full list):

- `--params <path>` — load parameters from a JSON or `.json.gz` file instead
  of the bundled defaults.
- `--seed <int>` — set the PRNG seed (default `20240930`).
- `--viz` / `--pdf` — display matplotlib visualisations or write a PDF.
- `--outdir <path>` — directory for any HDF5 or PDF outputs.
- `--over key:value` — override a single parameter, repeatable.
- `--loglevel {DEBUG,INFO,WARNING,ERROR,CRITICAL}` — verbosity.
- `-q` / `--quiet` — suppress the per-tick progress bar.

### From Python

[`run_model`][laser.cholera.metapop.model.run_model] returns the populated
model object:

```python
from laser.cholera.metapop.model import run_model

# Use the bundled defaults
model = run_model(None)
```

After `run_model` returns, the per-tick compartment state is on
`model.people` (e.g., `model.people.S`) and the per-patch derived quantities
are on `model.patches`. The final-tick model log-likelihood (when enabled
via the `calc_likelihood` parameter) is on `model.log_likelihood`.

## Where to look next

- **Tutorials** — [first run](tutorials/first-run.md),
  [single-location](tutorials/single-location.md),
  [multi-location country](tutorials/multi-location-country.md).
- **How-to guides** — [override parameters](how-to/override-parameters.md),
  [enable vaccination](how-to/enable-vaccination.md),
  [configure mobility](how-to/configure-mobility.md),
  [enable seasonality](how-to/enable-seasonality.md),
  [calibrate and score](how-to/calibrate-and-score.md),
  [interoperate with MOSAIC](how-to/interoperate-with-mosaic.md).
- **Reference** — [parameter index](reference/parameters/index.md),
  [API reference](reference/index.md).
- **Configurations** — [single-location](configurations/single-location.md),
  [Mozambique adm-2](configurations/multi-admin.md),
  [SSA baseline](configurations/ssa-baseline.md).
- **Explanation** — [model overview](explanation/model-overview.md),
  [transmission](explanation/transmission.md),
  [mobility](explanation/mobility.md),
  [reporting and likelihood](explanation/reporting-and-likelihood.md).

## Cross-checking against the upstream R reference

The R implementations the Python port was translated from live under
`reference/` at the repo root (not packaged into the sdist). They are useful
for sanity-checking the Python behaviour when something looks off.

### One-time setup

Install `testthat` and the upstream `MOSAIC` R package:

```bash
Rscript -e 'install.packages("testthat")'
Rscript -e 'remotes::install_github("InstituteforDiseaseModeling/MOSAIC-pkg")'
```

If `MOSAIC` is already on your R library path, you can skip the second
command. Verify with:

```bash
Rscript -e 'library(MOSAIC); packageVersion("MOSAIC")'
```

### Run the R test files

The R tests are free-standing `testthat` files; invoke them directly:

```bash
Rscript -e 'testthat::test_file("reference/test_calc_model_likelihood.R")'
```

Run from the repo root so the relative path resolves.

!!! warning "Caveats"
    - `MOSAIC` must be recent enough to include any upstream commits you want
      to cross-check against (e.g., the in-window peak filter).
    - R's `set.seed(123)` does not produce the same RNG stream as Python's
      `np.random.default_rng(123)`. Compare structural properties (finite,
      ordering, inequality) across the two, not exact numerical values — the
      Python test files already follow that convention.
