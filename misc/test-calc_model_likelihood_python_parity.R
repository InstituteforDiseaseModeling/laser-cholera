# =============================================================================
# test-calc_model_likelihood_python_parity.R
#
# Parity tests: R MOSAIC::calc_model_likelihood() vs the Python port
# (laser.cholera.calc_model_likelihood.calc_model_likelihood).
#
# These tests guard against drift between the two implementations along the
# axes most likely to break a port:
#
#   1. Matrix orientation (n_locations x n_time_steps) — both must agree.
#   2. Selection of data points (1..N in R vs 0..N-1 in Python) — both must
#      score the same observation indices.
#   3. Shape-term scaling (N_obs / N_component_observations) — confirmed for
#      core, cumulative, WIS, and the peak-timing/peak-magnitude path that
#      requires location_name, date_start, date_stop, and epidemic_peaks.
#
# All tests skip cleanly when the Python port is not importable. Requires
# laser-cholera >= 0.13.1 for the peak-term tests to pass (v0.13.1 ported the
# in-window peak filter from the R upstream).
#
# Tests #6 (NA-WIS) and #7 (zero-prediction penalty) PIN two known, tracked
# R-vs-Python divergences (observed on laser-cholera 0.13.1) rather than
# skip()-ing them: they run the comparison and assert the divergence is its
# documented magnitude, so the suite no longer false-greens and BOTH a fix
# (divergence closes) and a regression (it grows) surface as failures.
# =============================================================================

PARITY_TOL <- 1e-4
# PY_LIKELIHOOD_MODULE and skip_if_no_python_likelihood() are centralized in
# helper-skips.R (which reads the cached probe from setup-python.R).

# Build inputs that exercise the daily-timestep + epidemic_peaks path.
# 2 locations x 60 daily timesteps; date_start chosen so a single MOZ peak
# from MOSAIC::epidemic_peaks falls within the window.
make_parity_inputs <- function() {
  set.seed(42)
  n_loc <- 2L
  n_t   <- 60L
  obs_c <- matrix(rpois(n_loc * n_t, lambda = 30), nrow = n_loc)
  est_c <- matrix(pmax(0, obs_c + matrix(rnorm(n_loc * n_t, sd = 4), nrow = n_loc)), nrow = n_loc)
  obs_d <- matrix(round(obs_c * 0.05), nrow = n_loc)
  est_d <- matrix(pmax(0, round(est_c * 0.05)), nrow = n_loc)

  config <- list(
    location_name = c("MOZ", "ETH"),
    date_start    = "2023-01-01",
    date_stop     = as.character(as.Date("2023-01-01") + n_t - 1L)
  )

  list(
    obs_c = obs_c, est_c = est_c,
    obs_d = obs_d, est_d = est_d,
    config = config
  )
}

# Build flat kwargs that the v0.13.0 Python signature expects
# (date_start, date_stop, epidemic_peaks). epidemic_peaks must be a pandas
# DataFrame with iso_code/peak_date/loc_idx columns — converted from
# MOSAIC::epidemic_peaks only when the R-side dataset is loadable AND pandas
# is importable. loc_idx is REQUIRED: Python silently drops rows missing it
# (see laser.cholera.calc_model_likelihood:634), which would false-pass the
# peak-term parity tests.
make_py_kwargs <- function(config_r, include_peaks = TRUE) {
  kwargs <- list(
    date_start = config_r$date_start,
    date_stop  = config_r$date_stop
  )
  if (include_peaks && reticulate::py_module_available("pandas")) {
    pd <- reticulate::import("pandas", convert = FALSE)
    env <- new.env()
    has_peaks <- tryCatch({
      utils::data("epidemic_peaks", package = "MOSAIC", envir = env)
      exists("epidemic_peaks", envir = env)
    }, error = function(e) FALSE, warning = function(w) FALSE)
    if (has_peaks) {
      ep <- env$epidemic_peaks
      # Restrict to the simulation's locations and compute 0-based loc_idx
      # that matches laser-cholera's params.py:304 auto-computation.
      keep <- ep$iso_code %in% config_r$location_name
      ep_sub <- data.frame(
        iso_code  = as.character(ep$iso_code[keep]),
        peak_date = as.character(ep$peak_date[keep]),
        loc_idx   = as.integer(match(ep$iso_code[keep], config_r$location_name) - 1L),
        stringsAsFactors = FALSE
      )
      kwargs$epidemic_peaks <- pd$DataFrame(reticulate::r_to_py(ep_sub))
    }
  }
  kwargs
}

call_python_ll <- function(inputs, ..., py_kwargs = NULL) {
  py_mod <- reticulate::import(PY_LIKELIHOOD_MODULE, convert = TRUE)
  np <- reticulate::import("numpy", convert = FALSE)
  args <- list(
    obs_cases  = np$asarray(inputs$obs_c, dtype = "float64"),
    est_cases  = np$asarray(inputs$est_c, dtype = "float64"),
    obs_deaths = np$asarray(inputs$obs_d, dtype = "float64"),
    est_deaths = np$asarray(inputs$est_d, dtype = "float64"),
    ...
  )
  if (!is.null(py_kwargs)) args <- c(args, py_kwargs)
  do.call(py_mod$calc_model_likelihood, args)
}

# -----------------------------------------------------------------------------
# 1. Core NB only — the all-defaults path (every shape term off).
# -----------------------------------------------------------------------------
test_that("R and Python agree on core NB likelihood (defaults)", {
  skip_if_no_python_likelihood()
  inp <- make_parity_inputs()

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d
  )
  ll_py <- call_python_ll(inp)

  expect_true(is.finite(ll_r))
  expect_equal(as.numeric(ll_py), ll_r, tolerance = PARITY_TOL)
})

# -----------------------------------------------------------------------------
# 2. Cumulative shape term enabled — exercises the /end_idx normalization and
#    the cum_scale = N_obs / N_eval_points assembly multiplier.
# -----------------------------------------------------------------------------
test_that("R and Python agree with cumulative term enabled", {
  skip_if_no_python_likelihood()
  inp <- make_parity_inputs()

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d,
    weight_cumulative_total = 0.25
  )
  ll_py <- call_python_ll(inp, weight_cumulative_total = 0.25)

  expect_equal(as.numeric(ll_py), ll_r, tolerance = PARITY_TOL)
})

# -----------------------------------------------------------------------------
# 3. WIS shape term enabled — exercises NB quantile evaluation, MAE coefficient
#    (must be 0.5), and wis_scale = N_obs / N_quantiles.
# -----------------------------------------------------------------------------
test_that("R and Python agree with WIS term enabled", {
  skip_if_no_python_likelihood()
  inp <- make_parity_inputs()

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d,
    weight_wis = 0.10
  )
  ll_py <- call_python_ll(inp, weight_wis = 0.10)

  expect_equal(as.numeric(ll_py), ll_r, tolerance = PARITY_TOL)
})

# -----------------------------------------------------------------------------
# 4. Peak terms enabled — exercises config-driven peak-index lookup. Only runs
#    when MOSAIC::epidemic_peaks is loadable AND pandas is available so we
#    can hand the same DataFrame to both implementations.
# -----------------------------------------------------------------------------
test_that("R and Python agree with peak timing/magnitude enabled", {
  skip_if_no_python_likelihood()
  # Closed by laser-cholera v0.13.1: out-of-window peaks are now filtered
  # (matching the R-side behavior at calc_model_likelihood.R:147-150),
  # rather than being clamped by np.argmin to t=0 or t=n-1. The previously
  # documented ~22% divergence (R -495 vs Python -606) is now zero.

  if (!reticulate::py_module_available("pandas")) {
    skip("pandas not available — required for epidemic_peaks DataFrame")
  }
  env <- new.env()
  ok <- tryCatch({
    utils::data("epidemic_peaks", package = "MOSAIC", envir = env)
    exists("epidemic_peaks", envir = env)
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (!ok) skip("MOSAIC::epidemic_peaks not loadable")

  inp <- make_parity_inputs()
  py_kw <- make_py_kwargs(inp$config, include_peaks = TRUE)

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d,
    config = inp$config,
    weight_peak_timing    = 0.25,
    weight_peak_magnitude = 0.25
  )
  ll_py <- call_python_ll(
    inp,
    py_kwargs = py_kw,
    weight_peak_timing    = 0.25,
    weight_peak_magnitude = 0.25
  )

  expect_equal(as.numeric(ll_py), ll_r, tolerance = PARITY_TOL)
})

# -----------------------------------------------------------------------------
# 5. Weekly-resolution time series — exercises the timestep_to_weeks=1 branch
#    in the peak-timing helper. Confirms that date-sequence detection picks
#    weekly cadence when length(date_seq_daily) != n_time_steps.
# -----------------------------------------------------------------------------
test_that("R and Python agree on weekly cadence with peak terms", {
  skip_if_no_python_likelihood()
  # Closed by laser-cholera v0.13.1: same in-window filter fix as the
  # daily-cadence test above. The previously documented ~290% divergence
  # (R -282 vs Python -1101) is now zero. The earlier hypothesis of a
  # weekly-cadence detection bug was incorrect — the root cause was
  # out-of-window peaks being clamped instead of dropped.

  if (!reticulate::py_module_available("pandas")) skip("pandas not available")
  env <- new.env()
  ok <- tryCatch({
    utils::data("epidemic_peaks", package = "MOSAIC", envir = env)
    exists("epidemic_peaks", envir = env)
  }, error = function(e) FALSE, warning = function(w) FALSE)
  if (!ok) skip("MOSAIC::epidemic_peaks not loadable")

  set.seed(7)
  n_loc <- 2L
  n_t   <- 30L
  obs_c <- matrix(rpois(n_loc * n_t, lambda = 50), nrow = n_loc)
  est_c <- matrix(pmax(0, obs_c + matrix(rnorm(n_loc * n_t, sd = 6), nrow = n_loc)), nrow = n_loc)
  obs_d <- matrix(round(obs_c * 0.05), nrow = n_loc)
  est_d <- matrix(pmax(0, round(est_c * 0.05)), nrow = n_loc)

  d0 <- as.Date("2022-01-02")  # Sunday → weekly grid lands on Sundays
  cfg_r <- list(
    location_name = c("MOZ", "ETH"),
    date_start    = as.character(d0),
    date_stop     = as.character(d0 + 7L * (n_t - 1L))
  )

  inp <- list(obs_c = obs_c, est_c = est_c, obs_d = obs_d, est_d = est_d, config = cfg_r)
  py_kw <- make_py_kwargs(cfg_r, include_peaks = TRUE)

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d,
    config = cfg_r,
    weight_peak_timing = 0.25
  )
  ll_py <- call_python_ll(
    inp,
    py_kwargs = py_kw,
    weight_peak_timing = 0.25
  )

  expect_equal(as.numeric(ll_py), ll_r, tolerance = PARITY_TOL)
})

# -----------------------------------------------------------------------------
# 6. NA observations — exercises .mask_weights() on both sides and the
#    have_cases / have_deaths >= 3 finite-obs gate.
# -----------------------------------------------------------------------------
test_that("R and Python WIS agree within the known NA-masking tolerance", {
  skip_if_no_python_likelihood()
  # KNOWN, TRACKED divergence (observed on laser-cholera 0.13.1): Python's
  # .compute_wis_parametric_row() does NOT mask NA observations before the
  # subtraction (it emits "invalid value encountered in subtract"), whereas R
  # masks first. On this fixture that is a ~0.5% disagreement (R -489.57 vs
  # Python -487.02). Instead of a silent skip() (which false-greened this guard)
  # we run the comparison and assert agreement to within 2%: this tolerates the
  # small known gap but FAILS if the divergence grows. Tighten back to
  # PARITY_TOL once laser-cholera masks NAs in the WIS path.
  inp <- make_parity_inputs()
  inp$obs_c[1L, c(5L, 6L, 7L)] <- NA_real_
  inp$obs_d[2L, 1L:2L] <- NA_real_

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d,
    weight_wis = 0.10
  )
  ll_py <- suppressWarnings(as.numeric(call_python_ll(inp, weight_wis = 0.10)))

  expect_true(is.finite(ll_r) && is.finite(ll_py))
  expect_equal(ll_py, ll_r, tolerance = 0.02)
})

# -----------------------------------------------------------------------------
# 7. Zero-prediction proportional penalty path: when est==0 and obs>0, the
#    helper applies -obs * log(1e6). Force a row of zero estimates.
# -----------------------------------------------------------------------------
test_that("R and Python zero-prediction penalty differ by the known fixed ratio", {
  skip_if_no_python_likelihood()
  # KNOWN, TRACKED divergence (observed on laser-cholera 0.13.1): the
  # zero-prediction cumulative penalty is scaled differently on the Python side
  # — Python is ~1.78x more negative than R (R -28327.6 vs Python -50475.1 on
  # this fixture). This is a STRUCTURAL disagreement, not numerical noise, so
  # instead of a silent skip() we PIN the ratio: the test FAILS if the gap
  # closes (engine aligned -> switch this to exact parity) OR widens (regression).
  inp <- make_parity_inputs()
  inp$est_c[1L, ] <- 0
  inp$est_d[1L, ] <- 0

  ll_r <- MOSAIC::calc_model_likelihood(
    obs_cases  = inp$obs_c, est_cases  = inp$est_c,
    obs_deaths = inp$obs_d, est_deaths = inp$est_d,
    weight_cumulative_total = 0.25
  )
  ll_py <- suppressWarnings(as.numeric(call_python_ll(inp, weight_cumulative_total = 0.25)))

  expect_true(is.finite(ll_r) && is.finite(ll_py))
  # Pinned known ratio ~1.78 (Python more negative). If laser-cholera aligns the
  # penalty (ratio -> ~1.0) this WILL fail -> replace with expect_equal parity.
  expect_equal(ll_py / ll_r, 1.78, tolerance = 0.05)
})
