# =============================================================================
# R-vs-Python parity tests for calc_model_likelihood.
#
# Each test:
#   1. Builds an R-side fixture (matrices, optional vectors / matrices /
#      data.frames).
#   2. Calls MOSAIC::calc_model_likelihood (the canonical R implementation,
#      see misc/calc_model_likelihood.R).
#   3. Calls the Python laser.cholera implementation via reticulate
#      (the `py_calc_ll` helper in helper-setup.R).
#   4. Asserts the two scalars agree within the documented tolerance.
#
# Coverage matrix:
#
#   NB core only (acceptance criterion #7, ~1e-8):
#     - base case (small fixture, all defaults)
#     - single location
#     - NaN observations
#     - non-uniform weights_location
#     - non-uniform weights_time
#     - both location + time weights
#     - weight_cases / weight_deaths multipliers
#     - larger fixture (5 x 20)
#     - custom nb_k_min_cases / nb_k_min_deaths
#
#   Per-observation confidence weights (issue #86, ~1e-8):
#     - all-ones weights_obs_cases (must be byte-identical to NULL)
#     - mid-confidence flat (0.5) weights
#     - heterogeneous per-cell weights
#     - NaN per-cell weights
#     - zero-row weights (channel drops to zero LL)
#     - weights_obs_deaths only
#     - both weights_obs_cases AND weights_obs_deaths
#     - per-cell weights crossed with non-uniform weights_time
#
#   Shape terms (~1e-5, looser because R / scipy use independent special-
#   function paths through dnbinom, dlnorm, pnorm, qnbinom, etc.):
#     - cumulative-total only
#     - WIS only
#     - custom wis_quantiles
#     - custom cumulative_timepoints
#     - cumulative + WIS together
#
#   The peak-timing and peak-magnitude shape terms are NOT cross-checked here:
#   R's MOSAIC takes a `config` list (with location_name / date_start /
#   date_stop / epidemic_peaks) while Python takes `epidemic_peaks` /
#   `date_start` / `date_stop` directly. The two interfaces would need a
#   tailored fixture translator to be apples-to-apples; that is best done as
#   a follow-up once the obs-weights port is locked in. The same applies in
#   reverse: tests at the Python level already cover the peak terms via the
#   existing `tests/test_calc_model_likelihood.py` suite, and tests on the R
#   side already cover them via MOSAIC's own testthat files.
# =============================================================================


# ---------------------------------------------------------------------------
# NB core only
# ---------------------------------------------------------------------------

testthat::test_that("R-vs-Python: NB core base case (2x5, all defaults)", {
  skip_if_old_mosaic()
  obs_c <- matrix(c(3, 8, 12, 6, 2,
                    1, 4, 7, 3, 0), nrow = 2, byrow = TRUE)
  est_c <- matrix(c(2, 7, 10, 5, 3,
                    1, 5, 6, 4, 1), nrow = 2, byrow = TRUE)
  obs_d <- matrix(0, 2, 5)
  est_d <- matrix(0, 2, 5)

  ll_r  <- MOSAIC::calc_model_likelihood(obs_c, est_c, obs_d, est_d)
  ll_py <- py_calc_ll(obs_c, est_c, obs_d, est_d)

  expect_ll_match(ll_r, ll_py, "base case")
})

testthat::test_that("R-vs-Python: NB core single-location fixture", {
  skip_if_old_mosaic()
  obs_c <- matrix(c(3, 8, 12, 6, 2, 1, 0, 0, 1, 4), nrow = 1)
  est_c <- matrix(c(2, 7, 10, 5, 3, 1, 1, 1, 2, 5), nrow = 1)
  obs_d <- matrix(c(0, 1, 2, 1, 0, 0, 0, 0, 0, 0), nrow = 1)
  est_d <- matrix(c(0, 1, 1, 1, 0, 0, 0, 0, 1, 1), nrow = 1)

  ll_r  <- MOSAIC::calc_model_likelihood(obs_c, est_c, obs_d, est_d)
  ll_py <- py_calc_ll(obs_c, est_c, obs_d, est_d)

  expect_ll_match(ll_r, ll_py, "single location")
})

testthat::test_that("R-vs-Python: NB core with NaN observations", {
  skip_if_old_mosaic()
  obs_c <- matrix(c(0, 5, NA, 9, 2,
                    NA, 1, 4, 7, NA), nrow = 2, byrow = TRUE)
  est_c <- matrix(c(1, 4, 3, 8, 3,
                    2, 1, 5, 6, 4), nrow = 2, byrow = TRUE)
  obs_d <- matrix(0, 2, 5)
  est_d <- matrix(0, 2, 5)

  ll_r  <- MOSAIC::calc_model_likelihood(obs_c, est_c, obs_d, est_d)
  ll_py <- py_calc_ll(obs_c, est_c, obs_d, est_d)

  expect_ll_match(ll_r, ll_py, "NaN observations")
})

testthat::test_that("R-vs-Python: NB core with non-uniform weights_location", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()
  wloc <- c(0.5, 1.0, 1.5, 2.0, 0.25)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_location = wloc
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_location = wloc
  )

  expect_ll_match(ll_r, ll_py, "non-uniform weights_location")
})

testthat::test_that("R-vs-Python: NB core with non-uniform weights_time", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()
  wt <- c(0.1, 0.5, 1.0, 1.0, 1.0, 1.5, 1.5, 1.0, 0.5, 0.25)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_time = wt
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_time = wt
  )

  expect_ll_match(ll_r, ll_py, "non-uniform weights_time")
})

testthat::test_that("R-vs-Python: NB core with both weights_location and weights_time", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()
  wloc <- c(0.5, 1.0, 1.5, 2.0, 0.25)
  wt   <- seq(0.5, 1.5, length.out = fx$n_steps)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_location = wloc, weights_time = wt
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_location = wloc, weights_time = wt
  )

  expect_ll_match(ll_r, ll_py, "both location + time weights")
})

testthat::test_that("R-vs-Python: weight_cases and weight_deaths multipliers", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cases = 2.0, weight_deaths = 0.5
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cases = 2.0, weight_deaths = 0.5
  )

  expect_ll_match(ll_r, ll_py, "weight_cases / weight_deaths multipliers")
})

testthat::test_that("R-vs-Python: larger fixture (5x20)", {
  skip_if_old_mosaic()
  set.seed(7)
  obs_c <- matrix(rpois(5 * 20, lambda = 6), nrow = 5)
  est_c <- matrix(rpois(5 * 20, lambda = 5), nrow = 5)
  obs_d <- matrix(rpois(5 * 20, lambda = 2), nrow = 5)
  est_d <- matrix(rpois(5 * 20, lambda = 2), nrow = 5)

  ll_r  <- MOSAIC::calc_model_likelihood(obs_c, est_c, obs_d, est_d)
  ll_py <- py_calc_ll(obs_c, est_c, obs_d, est_d)

  expect_ll_match(ll_r, ll_py, "larger 5x20 fixture")
})

testthat::test_that("R-vs-Python: custom nb_k_min_cases / nb_k_min_deaths", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    nb_k_min_cases = 10, nb_k_min_deaths = 5
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    nb_k_min_cases = 10, nb_k_min_deaths = 5
  )

  expect_ll_match(ll_r, ll_py, "custom nb_k_min")
})


# ---------------------------------------------------------------------------
# Per-observation confidence weights (issue #86 cross-check)
# ---------------------------------------------------------------------------

testthat::test_that("R-vs-Python: weights_obs_cases all-ones (trivial path)", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  wobs <- matrix(1.0, fx$n_locs, fx$n_steps)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )

  expect_ll_match(ll_r, ll_py, "weights_obs_cases all-ones")
})

testthat::test_that("R-vs-Python: weights_obs_cases mid-confidence (0.5)", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  wobs <- matrix(0.5, fx$n_locs, fx$n_steps)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )

  expect_ll_match(ll_r, ll_py, "weights_obs_cases flat 0.5")
})

testthat::test_that("R-vs-Python: heterogeneous per-cell weights_obs_cases", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  set.seed(13)
  wobs <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.3, max = 1.0), nrow = fx$n_locs)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )

  expect_ll_match(ll_r, ll_py, "heterogeneous per-cell weights")
})

testthat::test_that("R-vs-Python: NaN entries in weights_obs_cases", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  wobs <- matrix(0.8, fx$n_locs, fx$n_steps)
  wobs[1, 2] <- NA_real_
  wobs[3, 5] <- NA_real_
  wobs[4, 9] <- NA_real_

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )

  expect_ll_match(ll_r, ll_py, "NaN entries in weights_obs_cases")
})

testthat::test_that("R-vs-Python: zero-row weights_obs_cases at one location", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  wobs <- matrix(0.9, fx$n_locs, fx$n_steps)
  wobs[2, ] <- 0   # zero-out location 2's cases channel entirely

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs
  )

  expect_ll_match(ll_r, ll_py, "zero-row weights_obs_cases")
})

testthat::test_that("R-vs-Python: weights_obs_deaths only (cases NULL)", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  set.seed(17)
  wobs_d <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.4, max = 1.0), nrow = fx$n_locs)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_deaths = wobs_d
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_deaths = wobs_d
  )

  expect_ll_match(ll_r, ll_py, "weights_obs_deaths only")
})

testthat::test_that("R-vs-Python: both weights_obs_cases and weights_obs_deaths", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  set.seed(19)
  wobs_c <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.3, max = 1.0), nrow = fx$n_locs)
  wobs_d <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.4, max = 1.0), nrow = fx$n_locs)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs_c, weights_obs_deaths = wobs_d
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_obs_cases = wobs_c, weights_obs_deaths = wobs_d
  )

  expect_ll_match(ll_r, ll_py, "both weights_obs_*")
})

testthat::test_that("R-vs-Python: per-cell weights crossed with non-uniform weights_time", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  wt <- seq(0.3, 1.7, length.out = fx$n_steps)
  set.seed(23)
  wobs_c <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.5, max = 1.0), nrow = fx$n_locs)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_time = wt, weights_obs_cases = wobs_c
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weights_time = wt, weights_obs_cases = wobs_c
  )

  expect_ll_match(ll_r, ll_py, "per-cell + non-uniform weights_time")
})

testthat::test_that("R-vs-Python: documented-zero-heavy pattern (issue #86 scenario)", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  T <- 50L
  obs_c <- matrix(0, 1, T); obs_c[1, 1:5] <- c(3, 8, 12, 6, 2)
  est_c <- matrix(2, 1, T); est_c[1, 1:5] <- c(2, 7, 10, 5, 3)
  obs_d <- matrix(0, 1, T)
  est_d <- matrix(0, 1, T)
  wobs <- matrix(0.80, 1, T); wobs[1, 1:5] <- 0.95

  ll_r  <- MOSAIC::calc_model_likelihood(obs_c, est_c, obs_d, est_d, weights_obs_cases = wobs)
  ll_py <- py_calc_ll(obs_c, est_c, obs_d, est_d, weights_obs_cases = wobs)

  expect_ll_match(ll_r, ll_py, "documented-zero-heavy scenario")
})


# ---------------------------------------------------------------------------
# Shape terms (looser tolerance: ~1e-5 absolute on the total LL)
# ---------------------------------------------------------------------------

testthat::test_that("R-vs-Python: cumulative-total shape term only", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cumulative_total = 0.25
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cumulative_total = 0.25
  )

  expect_ll_match(ll_r, ll_py, "cumulative-total only", abs_tol = 1e-5, rel_tol = 1e-6)
})

testthat::test_that("R-vs-Python: WIS shape term only", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_wis = 0.25
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_wis = 0.25
  )

  expect_ll_match(ll_r, ll_py, "WIS only", abs_tol = 1e-5, rel_tol = 1e-6)
})

testthat::test_that("R-vs-Python: custom wis_quantiles", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()
  quants <- c(0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_wis = 0.5, wis_quantiles = quants
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_wis = 0.5, wis_quantiles = quants
  )

  expect_ll_match(ll_r, ll_py, "custom wis_quantiles", abs_tol = 1e-5, rel_tol = 1e-6)
})

testthat::test_that("R-vs-Python: custom cumulative_timepoints", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()
  cum_tps <- c(0.1, 0.3, 0.5, 0.7, 0.9, 1.0)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cumulative_total = 0.3, cumulative_timepoints = cum_tps
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cumulative_total = 0.3, cumulative_timepoints = cum_tps
  )

  expect_ll_match(ll_r, ll_py, "custom cumulative_timepoints", abs_tol = 1e-5, rel_tol = 1e-6)
})

testthat::test_that("R-vs-Python: cumulative + WIS together", {
  skip_if_old_mosaic()
  fx <- .build_fixture_5x10()

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cumulative_total = 0.25, weight_wis = 0.25
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cumulative_total = 0.25, weight_wis = 0.25
  )

  expect_ll_match(ll_r, ll_py, "cumulative + WIS", abs_tol = 1e-5, rel_tol = 1e-6)
})

testthat::test_that("R-vs-Python: all NB core + cumulative + WIS + obs-weights kitchen-sink", {
  skip_if_old_mosaic()
  skip_if_no_obs_weights()
  fx <- .build_fixture_5x10()
  set.seed(29)
  wloc <- c(0.5, 1.0, 1.5, 2.0, 0.25)
  wt <- seq(0.5, 1.5, length.out = fx$n_steps)
  wobs_c <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.5, max = 1.0), nrow = fx$n_locs)
  wobs_d <- matrix(runif(fx$n_locs * fx$n_steps, min = 0.6, max = 1.0), nrow = fx$n_locs)

  ll_r  <- MOSAIC::calc_model_likelihood(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cases = 1.5, weight_deaths = 0.75,
    weights_location = wloc, weights_time = wt,
    weights_obs_cases = wobs_c, weights_obs_deaths = wobs_d,
    weight_cumulative_total = 0.2, weight_wis = 0.2,
    nb_k_min_cases = 5, nb_k_min_deaths = 3
  )
  ll_py <- py_calc_ll(
    fx$obs_cases, fx$est_cases, fx$obs_deaths, fx$est_deaths,
    weight_cases = 1.5, weight_deaths = 0.75,
    weights_location = wloc, weights_time = wt,
    weights_obs_cases = wobs_c, weights_obs_deaths = wobs_d,
    weight_cumulative_total = 0.2, weight_wis = 0.2,
    nb_k_min_cases = 5, nb_k_min_deaths = 3
  )

  expect_ll_match(ll_r, ll_py, "kitchen-sink", abs_tol = 1e-5, rel_tol = 1e-6)
})
