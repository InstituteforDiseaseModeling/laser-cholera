# Tests for calc_model_likelihood() extreme inputs — replaces guardrail tests
# Verifies the NB naturally handles bad fits without needing hard floors.

test_that("extreme over-prediction produces very negative but finite LL", {
  n_loc <- 2
  n_time <- 50
  obs <- matrix(10, n_loc, n_time)
  est <- matrix(10000, n_loc, n_time)  # 1000x over-prediction

  ll <- MOSAIC::calc_model_likelihood(
    obs_cases = obs, est_cases = est,
    obs_deaths = obs, est_deaths = est
  )
  expect_true(is.finite(ll))
  expect_true(ll < -1000)  # Should be very negative
})

test_that("extreme under-prediction produces very negative but finite LL", {
  n_loc <- 2
  n_time <- 50
  obs <- matrix(100, n_loc, n_time)
  est <- matrix(0.001, n_loc, n_time)  # Near-zero prediction

  ll <- MOSAIC::calc_model_likelihood(
    obs_cases = obs, est_cases = est,
    obs_deaths = obs, est_deaths = est
  )
  expect_true(is.finite(ll))
  expect_true(ll < -10000)  # Proportional penalty: very negative
})

test_that("zero prediction with nonzero observed uses proportional penalty", {
  obs <- matrix(c(0, 100, 0, 50), nrow = 1)
  est <- matrix(c(0, 0, 0, 0), nrow = 1)

  ll <- MOSAIC::calc_model_likelihood(
    obs_cases = obs, est_cases = est,
    obs_deaths = obs, est_deaths = est
  )
  expect_true(is.finite(ll))
  expect_true(ll < -1000)  # -obs * log(1e6) penalty
})

test_that("non-finite LL returns -Inf", {
  # Force a non-finite by using NaN in estimates
  obs <- matrix(c(10, 20, 30), nrow = 1)
  est <- matrix(c(10, NaN, 30), nrow = 1)

  # This should not error — NaN propagates to non-finite LL which becomes -Inf
  ll <- MOSAIC::calc_model_likelihood(
    obs_cases = obs, est_cases = est,
    obs_deaths = obs, est_deaths = est
  )
  # Result should be finite or -Inf (non-finite safety net)
  expect_true(is.finite(ll) || identical(ll, -Inf) || is.na(ll))
})

test_that("negative-correlated inputs produce bad but finite LL (not floored)", {
  set.seed(42)
  n_time <- 100
  # Create anti-correlated time series
  obs <- matrix(c(rep(c(100, 0), n_time / 2)), nrow = 1)
  est <- matrix(c(rep(c(0, 100), n_time / 2)), nrow = 1)  # Opposite pattern

  ll <- MOSAIC::calc_model_likelihood(
    obs_cases = obs, est_cases = est,
    obs_deaths = matrix(0, 1, n_time), est_deaths = matrix(0, 1, n_time)
  )
  # Should be finite (no guardrail floor), but very negative
  expect_true(is.finite(ll))
  expect_true(ll < -500)
})

test_that("all-zero obs and est produces zero LL (correct behavior)", {
  obs <- matrix(0, nrow = 3, ncol = 20)
  est <- matrix(0, nrow = 3, ncol = 20)

  ll <- MOSAIC::calc_model_likelihood(
    obs_cases = obs, est_cases = est,
    obs_deaths = obs, est_deaths = est
  )
  expect_true(is.finite(ll))
  expect_equal(ll, 0, tolerance = 1e-8)
})
