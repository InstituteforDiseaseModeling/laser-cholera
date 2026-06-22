# =============================================================================
# Common setup for R-vs-Python parity tests of `calc_model_likelihood`.
#
# Loaded automatically by `testthat::test_dir()` before any `test-*.R` files
# (per the `helper-*.R` naming convention).
#
# Responsibilities:
#   1. Hard-fail with an actionable message if MOSAIC / reticulate / testthat
#      is missing.
#   2. Point `reticulate` at the project's `.venv/bin/python3` when present,
#      so the Python `laser.cholera` import resolves the same way `pytest`
#      does for the rest of the test suite.
#   3. Print MOSAIC + laser-cholera + R version banner once, so any failure
#      report identifies the exact stack under test.
#   4. Expose two helpers used by every test file:
#        - `py_calc_ll(...)`: call Python `calc_model_likelihood` with R-side
#          matrices / data.frames, returning the scalar log-likelihood.
#        - `expect_ll_match(ll_r, ll_py, ...)`: assert the two scalars agree
#          within the tolerance documented in the acceptance criteria for
#          issue #86 (~1e-8 on the total LL for the NB core; slightly looser
#          for the optional shape terms because R and scipy use independent
#          special-function paths).
# =============================================================================

if (!requireNamespace("MOSAIC", quietly = TRUE)) {
  stop(
    "MOSAIC package not installed. Install with:\n",
    "  Rscript -e 'remotes::install_github(\"InstituteforDiseaseModeling/MOSAIC\")'"
  )
}
if (!requireNamespace("reticulate", quietly = TRUE)) {
  stop(
    "reticulate package not installed. Install with:\n",
    "  Rscript -e 'install.packages(\"reticulate\")'"
  )
}

# -- Point reticulate at the project's editable venv --------------------------
# `RETICULATE_PYTHON` is honoured AT LOAD TIME by `library(reticulate)` and
# OVERRIDES any later `reticulate::use_python()` call. If a user has it set
# globally (e.g. to an `r-mosaic` virtualenv with a stale, non-editable
# laser-cholera snapshot), `use_python(...)` is silently ignored and the
# tests end up calling the wrong Python module, producing both "unused
# argument" errors and silent semantic drift.
#
# Override the env var BEFORE `library(reticulate)` is loaded so the project's
# editable venv wins. Same defense-in-depth as `tests/R/run_tests.sh`.
.repo_root <- normalizePath(file.path(getwd(), "..", ".."), mustWork = FALSE)
if (!dir.exists(.repo_root) || !file.exists(file.path(.repo_root, "pyproject.toml"))) {
  # Fallback: walk up from the source file location (works when invoked via
  # `Rscript -e 'testthat::test_dir(\"tests/R\")'` from the repo root).
  .repo_root <- normalizePath(file.path(dirname(sys.frame(1)$ofile), "..", ".."), mustWork = FALSE)
}
.venv_python <- file.path(.repo_root, ".venv", "bin", "python3")
if (file.exists(.venv_python)) {
  Sys.setenv(RETICULATE_PYTHON = .venv_python)
}

suppressPackageStartupMessages({
  library(MOSAIC)
  library(reticulate)
})

if (file.exists(.venv_python)) {
  # Belt-and-braces — even with the env var set, call `use_python` so the
  # config object reflects the actual interpreter (some reticulate paths
  # check this directly rather than re-reading the env var).
  reticulate::use_python(.venv_python, required = TRUE)
}

# -- Import Python modules ----------------------------------------------------
.lc_likelihood <- reticulate::import("laser.cholera.calc_model_likelihood")
.np <- reticulate::import("numpy")
.pd <- reticulate::import("pandas")
.datetime_mod <- reticulate::import("datetime")

# -- Version banner -----------------------------------------------------------
# -- MOSAIC minimum version required for parity ------------------------------
# The Python `calc_model_likelihood` tracks the MOSAIC reference at v0.45.3
# (commit `7a265b1d`, see misc/calc_model_likelihood.R). Older MOSAIC
# installs use a different assembly formula and lack the `weights_obs_cases`
# / `weights_obs_deaths` arguments entirely. Tests skip cleanly on older
# installs rather than fail noisily.
.MIN_MOSAIC_VERSION <- "0.45.3"
.installed_mosaic_version <- as.character(packageVersion("MOSAIC"))
.mosaic_ok <- utils::compareVersion(.installed_mosaic_version, .MIN_MOSAIC_VERSION) >= 0
.mosaic_has_obs_weights <- "weights_obs_cases" %in% names(formals(MOSAIC::calc_model_likelihood))

cat("\n")
cat("=== R-vs-Python calc_model_likelihood parity ===\n")
cat(sprintf("  R version           : %s\n", R.version.string))
cat(sprintf("  MOSAIC version      : %s (required >= %s)\n",
            .installed_mosaic_version, .MIN_MOSAIC_VERSION))
cat(sprintf("  MOSAIC has obs-weights: %s\n", .mosaic_has_obs_weights))
.lc_version <- tryCatch(
  reticulate::py_run_string(
    "import laser.cholera as _lc\n_v = getattr(_lc, '__version__', 'unknown')"
  )$`_v`,
  error = function(e) "unknown"
)
cat(sprintf("  laser-cholera version: %s\n", .lc_version))
cat(sprintf("  python executable    : %s\n", reticulate::py_config()$python))
cat(sprintf("  numpy version        : %s\n", as.character(.np$`__version__`)))
if (!.mosaic_ok) {
  cat("\n")
  cat(sprintf("  !! MOSAIC %s is older than the v%s reference the Python\n",
              .installed_mosaic_version, .MIN_MOSAIC_VERSION))
  cat("     `calc_model_likelihood` tracks. All parity tests will SKIP.\n")
  cat("     Update with:\n")
  cat("       Rscript -e 'remotes::install_github(\"InstituteforDiseaseModeling/MOSAIC\")'\n")
}
cat("\n")

# Per-test skip guard: every parity test starts with `skip_if_old_mosaic()`
# so the suite reports cleanly when MOSAIC is too old rather than producing
# misleading numerical failures.
skip_if_old_mosaic <- function() {
  if (!.mosaic_ok) {
    testthat::skip(sprintf(
      "MOSAIC %s < %s — install the v%s+ MOSAIC release to run the R-vs-Python parity check",
      .installed_mosaic_version, .MIN_MOSAIC_VERSION, .MIN_MOSAIC_VERSION
    ))
  }
}

# Some tests exercise the per-observation confidence weights, which were
# introduced in MOSAIC v0.45.3. The min-version guard above already implies
# the new args exist, but this explicit predicate makes the dependency
# obvious in each test body.
skip_if_no_obs_weights <- function() {
  if (!.mosaic_has_obs_weights) {
    testthat::skip(sprintf(
      "Installed MOSAIC %s does not have weights_obs_cases / weights_obs_deaths arguments",
      .installed_mosaic_version
    ))
  }
}

# -- Conversion helpers -------------------------------------------------------
# R matrices are column-major; numpy is row-major. `reticulate::r_to_py` on a
# matrix preserves shape and contents (it does NOT transpose); both backends
# read `obs[j, t]` with the same semantics.
.as_np_matrix <- function(m) {
  if (is.null(m)) return(NULL)
  .np$asarray(m, dtype = .np$float64)
}

# Convert an R Date / POSIXct / string to a Python `datetime.datetime`.
.as_py_datetime <- function(x) {
  if (is.null(x)) return(NULL)
  if (is.character(x)) {
    parts <- as.integer(strsplit(x, "-", fixed = TRUE)[[1]])
    return(.datetime_mod$datetime(parts[1], parts[2], parts[3]))
  }
  if (inherits(x, "Date") || inherits(x, "POSIXt")) {
    return(.datetime_mod$datetime(
      as.integer(format(x, "%Y")),
      as.integer(format(x, "%m")),
      as.integer(format(x, "%d"))
    ))
  }
  stop("Unsupported date type for Python conversion: ", class(x))
}

# Convert an R data.frame (epidemic_peaks) to a pandas DataFrame. The Python
# `calc_model_likelihood` reads `iso_code`, `peak_date`, and `loc_idx` columns;
# the R caller must supply all three explicitly (the JSON ingestion path that
# auto-augments `loc_idx` is not in play here).
.as_pd_dataframe <- function(df) {
  if (is.null(df)) return(NULL)
  reticulate::r_to_py(df)
}

# -- Python LL caller ---------------------------------------------------------
# Accepts the same R-side argument set as MOSAIC::calc_model_likelihood and
# routes everything through reticulate. Argument names map directly.
py_calc_ll <- function(obs_cases, est_cases, obs_deaths, est_deaths,
                       weight_cases = 1.0,
                       weight_deaths = 1.0,
                       weights_location = NULL,
                       weights_time = NULL,
                       weights_obs_cases = NULL,
                       weights_obs_deaths = NULL,
                       weight_peak_timing = 0,
                       weight_peak_magnitude = 0,
                       weight_cumulative_total = 0,
                       weight_wis = 0,
                       sigma_peak_time = 1,
                       sigma_peak_log = 0.5,
                       epidemic_peaks = NULL,
                       date_start = NULL,
                       date_stop = NULL,
                       wis_quantiles = c(0.025, 0.25, 0.5, 0.75, 0.975),
                       cumulative_timepoints = c(0.25, 0.5, 0.75, 1.0),
                       nb_k_min_cases = 3,
                       nb_k_min_deaths = 3) {
  kwargs <- list(
    obs_cases             = .as_np_matrix(obs_cases),
    est_cases             = .as_np_matrix(est_cases),
    obs_deaths            = .as_np_matrix(obs_deaths),
    est_deaths            = .as_np_matrix(est_deaths),
    weight_cases          = weight_cases,
    weight_deaths         = weight_deaths,
    weight_peak_timing    = weight_peak_timing,
    weight_peak_magnitude = weight_peak_magnitude,
    weight_cumulative_total = weight_cumulative_total,
    weight_wis            = weight_wis,
    sigma_peak_time       = sigma_peak_time,
    sigma_peak_log        = sigma_peak_log,
    wis_quantiles         = .np$asarray(wis_quantiles, dtype = .np$float64),
    cumulative_timepoints = .np$asarray(cumulative_timepoints, dtype = .np$float64),
    nb_k_min_cases        = nb_k_min_cases,
    nb_k_min_deaths       = nb_k_min_deaths
  )
  if (!is.null(weights_location)) {
    kwargs$weights_location <- .np$asarray(weights_location, dtype = .np$float64)
  }
  if (!is.null(weights_time)) {
    kwargs$weights_time <- .np$asarray(weights_time, dtype = .np$float64)
  }
  if (!is.null(weights_obs_cases)) {
    kwargs$weights_obs_cases <- .as_np_matrix(weights_obs_cases)
  }
  if (!is.null(weights_obs_deaths)) {
    kwargs$weights_obs_deaths <- .as_np_matrix(weights_obs_deaths)
  }
  if (!is.null(epidemic_peaks)) {
    kwargs$epidemic_peaks <- .as_pd_dataframe(epidemic_peaks)
  }
  if (!is.null(date_start)) {
    kwargs$date_start <- .as_py_datetime(date_start)
  }
  if (!is.null(date_stop)) {
    kwargs$date_stop <- .as_py_datetime(date_stop)
  }
  as.numeric(do.call(.lc_likelihood$calc_model_likelihood, kwargs))
}

# -- Comparison helper --------------------------------------------------------
# Issue #86 acceptance criterion #7 calls for ~1e-8 on the total LL for the
# NB-core path. Optional shape terms (peak / cumulative / WIS) involve
# independent special-function paths in R (`pnorm`, `dnbinom`, `dlnorm`) and
# scipy (`stats.norm`, `stats.nbinom`, `stats.lognorm`), which can diverge by
# a few ULPs across long sums; for those a relative tolerance of 1e-6 is more
# realistic. Tests should pass the most appropriate `abs_tol` / `rel_tol`.
expect_ll_match <- function(ll_r, ll_py, label = "",
                            abs_tol = 1e-8, rel_tol = 1e-6) {
  diff_info <- function() {
    sprintf("%sR = %.12g  Py = %.12g  abs = %.3g  rel = %.3g",
            if (nchar(label) > 0) paste0("[", label, "] ") else "",
            ll_r, ll_py,
            abs(ll_r - ll_py),
            abs(ll_r - ll_py) / max(abs(ll_r), 1e-12))
  }
  if (!is.finite(ll_r) || !is.finite(ll_py)) {
    testthat::expect_equal(ll_r, ll_py, info = diff_info())
    return(invisible())
  }
  abs_diff <- abs(ll_r - ll_py)
  rel_diff <- abs_diff / max(abs(ll_r), 1e-12)
  testthat::expect_true(
    abs_diff < abs_tol || rel_diff < rel_tol,
    info = diff_info()
  )
}

# Standard fixture builder used by several tests.
.build_fixture_5x10 <- function(seed = 42) {
  set.seed(seed)
  n_locs <- 5L
  n_steps <- 10L
  obs_cases <- matrix(rpois(n_locs * n_steps, lambda = 5), nrow = n_locs)
  est_cases <- matrix(rpois(n_locs * n_steps, lambda = 4), nrow = n_locs)
  obs_deaths <- matrix(rpois(n_locs * n_steps, lambda = 1), nrow = n_locs)
  est_deaths <- matrix(rpois(n_locs * n_steps, lambda = 1), nrow = n_locs)
  list(
    obs_cases = obs_cases, est_cases = est_cases,
    obs_deaths = obs_deaths, est_deaths = est_deaths,
    n_locs = n_locs, n_steps = n_steps
  )
}
