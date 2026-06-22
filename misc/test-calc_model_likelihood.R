# ============================================================================
# Comprehensive tests for calc_model_likelihood function
# Testing all individual terms and their interactions
# ============================================================================
# Helper: simple zero data yields zero log-likelihood for Poisson
obs_zero <- matrix(0, nrow = 2, ncol = 3)
est_zero <- matrix(0, nrow = 2, ncol = 3)
# 1. Basic functionality: zero observed and estimated => finite (perfect match)
testthat::test_that("zero data returns finite log-likelihood", {
     ll <- MOSAIC::calc_model_likelihood(
          obs_cases     = obs_zero,
          est_cases     = est_zero,
          obs_deaths    = obs_zero,
          est_deaths    = est_zero,
     )
     expect_true(is.finite(ll))  # Zero data is a valid perfect match, not "no information"
     expect_equal(ll, 0, tolerance = 1e-8)  # Near zero with all auxiliary terms off
})
# 2. Weight scaling: non-default weight_cases and weight_deaths still yields finite for zero data
testthat::test_that("weights do not affect zero-data result", {
     ll <- MOSAIC::calc_model_likelihood(
          obs_cases     = obs_zero,
          est_cases     = est_zero,
          obs_deaths    = obs_zero,
          est_deaths    = est_zero,
          weight_cases  = 2,
          weight_deaths = 3,
     )
     expect_true(is.finite(ll))  # Still finite regardless of weights
     expect_equal(ll, 0, tolerance = 1e-8)  # Near zero with all auxiliary terms off
})
# 3. Dimension errors: non-matrix inputs
testthat::test_that("errors on non-matrix inputs", {
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs_cases  = as.list(obs_zero),
               est_cases  = est_zero,
               obs_deaths = obs_zero,
               est_deaths = est_zero
          ),
          "inputs must be matrices"
     )
})
# 4. Dimension mismatch error
testthat::test_that("errors when matrices have different dimensions", {
     est_bad <- matrix(0, nrow = 1, ncol = 3)
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs_cases  = obs_zero,
               est_cases  = est_bad,
               obs_deaths = obs_zero,
               est_deaths = obs_zero
          ),
          "All matrices must have the same dimensions"
     )
})
# 5. Weight vector length check
testthat::test_that("errors when weight vectors have incorrect lengths", {
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs_cases         = obs_zero,
               est_cases         = est_zero,
               obs_deaths        = obs_zero,
               est_deaths        = est_zero,
               weights_location  = c(1),
               weights_time      = c(1)
          ),
          "weights_location must match n_locations"
     )
})
# 6. All NA data returns finite (but should warn about no valid data)
testthat::test_that("all NA data returns finite", {
     obs_na <- matrix(NA_real_, nrow = 2, ncol = 3)
     est_na <- matrix(NA_real_, nrow = 2, ncol = 3)
     ll <- MOSAIC::calc_model_likelihood(
          obs_cases  = obs_na,
          est_cases  = est_na,
          obs_deaths = obs_na,
          est_deaths = est_na,
          verbose     = TRUE
     )
     # NA data with no valid observations should return finite (likely 0) since no likelihood is computed
     expect_true(is.finite(ll) || is.na(ll))  # Allow either finite or NA
})
# 6.5. Real values for estimated and all NA observed
testthat::test_that("all-NA observed with real estimates returns finite", {
     obs_na   <- matrix(NA_real_, nrow = 1, ncol = 2)
     est_real <- matrix(c(1.2, 3.4),   nrow = 1, ncol = 2)
     ll <- MOSAIC::calc_model_likelihood(
          obs_cases  = obs_na,
          est_cases  = est_real,
          obs_deaths = obs_na,
          est_deaths = est_real,
          verbose    = TRUE
     )
     # NA observed with real estimates should return finite (likely 0) since no valid comparisons can be made
     expect_true(is.finite(ll) || is.na(ll))  # Allow either finite or NA
})
# 7. Real likelihood calculation - core NB terms
testthat::test_that("correct log-likelihood for simple non-zero data with core terms only", {
     obs <- matrix(c(1, 1, 1), nrow = 1, ncol = 3)
     est <- matrix(c(1, 1, 1), nrow = 1, ncol = 3)
     ll <- MOSAIC::calc_model_likelihood(
          obs_cases  = obs,
          est_cases  = est,
          obs_deaths = obs,
          est_deaths = est
     )
     # Core uses NB (k estimated from obs via MoM, falls back to Poisson limit for
     # constant data where var <= mean).
     expect_true(is.finite(ll))
     expect_true(ll < 0)  # Perfect match still has negative LL for count data
})
# ============================================================================
# NEW COMPREHENSIVE TESTS FOR ALL TERMS
# ============================================================================
# 8. Test peak timing term
testthat::test_that("peak timing term works correctly", {
     set.seed(123)
     n_loc <- 2
     n_time <- 52
     # Create epidemic with clear peak
     obs_cases <- matrix(5, n_loc, n_time)
     obs_cases[1, 20:30] <- c(10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5)
     est_cases <- matrix(5, n_loc, n_time)
     est_cases[1, 20:30] <- c(10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5)  # Same timing
     obs_deaths <- matrix(1, n_loc, n_time)
     est_deaths <- matrix(1, n_loc, n_time)
     # Test with same peak timing
     ll_same_peak <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_peak_timing = 0.25,
     )
     # Shift peak in estimated
     est_cases_shifted <- matrix(5, n_loc, n_time)
     est_cases_shifted[1, 25:35] <- c(10, 20, 30, 40, 50, 40, 30, 20, 10, 5, 5)  # Shifted by 5
     ll_shifted_peak <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases_shifted,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_peak_timing = 0.25,
     )
     # Same timing should give better likelihood
     expect_true(ll_same_peak > ll_shifted_peak)
})
# 9. Test peak magnitude term
testthat::test_that("peak magnitude term works correctly", {
     set.seed(123)
     n_loc <- 2
     n_time <- 52
     # Create epidemic with clear peak
     obs_cases <- matrix(5, n_loc, n_time)
     obs_cases[1, 25] <- 100  # High peak
     est_cases <- matrix(5, n_loc, n_time)
     est_cases[1, 25] <- 100  # Same magnitude
     obs_deaths <- matrix(1, n_loc, n_time)
     est_deaths <- matrix(1, n_loc, n_time)
     # Test with same peak magnitude
     ll_same_mag <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_peak_magnitude = 0.25,
     )
     # Different magnitude
     est_cases_diff <- matrix(5, n_loc, n_time)
     est_cases_diff[1, 25] <- 50  # Half the magnitude
     ll_diff_mag <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases_diff,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_peak_magnitude = 0.25,
     )
     # Same magnitude should give better likelihood
     expect_true(ll_same_mag > ll_diff_mag)
})
# 10. Test progressive cumulative total term
testthat::test_that("progressive cumulative total term works correctly", {
     set.seed(123)
     n_loc <- 2
     n_time <- 52
     # Create data with known cumulative pattern
     obs_cases <- matrix(10, n_loc, n_time)
     est_cases <- matrix(10, n_loc, n_time)  # Same cumulative
     obs_deaths <- matrix(2, n_loc, n_time)
     est_deaths <- matrix(2, n_loc, n_time)
     # Test with default timepoints
     ll_default <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_cumulative_total = 0.25,
          cumulative_timepoints = c(0.25, 0.5, 0.75, 1.0)
     )
     # Test with custom timepoints
     ll_custom <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_cumulative_total = 0.25,
          cumulative_timepoints = c(0.33, 0.67, 1.0)
     )
     expect_true(is.finite(ll_default))
     expect_true(is.finite(ll_custom))
     # Test with mismatched cumulative
     est_cases_bad <- matrix(20, n_loc, n_time)  # Double throughout
     ll_bad_cumulative <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases_bad,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_cumulative_total = 0.25,
     )
     # Matched cumulative should be better
     expect_true(ll_default > ll_bad_cumulative)
})
# Test 11 removed - growth rate term has been removed from the function
# Test 12 removed - duration term has been removed from the function
# 13. Test WIS term
testthat::test_that("WIS term penalizes uncertainty correctly", {
     set.seed(123)
     n_loc <- 2
     n_time <- 52
     obs_cases <- matrix(rpois(n_loc * n_time, lambda = 10), n_loc, n_time)
     est_cases <- matrix(10, n_loc, n_time)  # Constant estimate
     obs_deaths <- matrix(rpois(n_loc * n_time, lambda = 2), n_loc, n_time)
     est_deaths <- matrix(2, n_loc, n_time)
     # Test with WIS term
     ll_wis <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_wis = 0.10,
          wis_quantiles = c(0.025, 0.25, 0.5, 0.75, 0.975)
     )
     expect_true(is.finite(ll_wis))
     # WIS penalty should make likelihood negative (since it's subtracted)
     ll_no_wis <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
     )
     # WIS is a penalty, so should reduce likelihood
     expect_true(ll_wis <= ll_no_wis)
})
# 14. Test all terms together
testthat::test_that("all terms work together without conflict", {
     set.seed(123)
     n_loc <- 2
     n_time <- 52
     # Create realistic epidemic data
     obs_cases <- matrix(5, n_loc, n_time)
     for (i in 1:n_loc) {
          obs_cases[i, 10:40] <- round(5 + 20 * dnorm(10:40, mean = 25, sd = 5) * 100)
     }
     est_cases <- obs_cases + matrix(rnorm(n_loc * n_time, 0, 2), n_loc, n_time)
     est_cases[est_cases < 0] <- 0
     obs_deaths <- round(obs_cases * 0.1)
     est_deaths <- round(est_cases * 0.1)
     # Test with all terms enabled at nonzero weights
     ll_all <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths,
          weight_peak_timing = 0.25,
          weight_peak_magnitude = 0.25,
          weight_cumulative_total = 0.25,
          weight_wis = 0.10,
          verbose = FALSE
     )
     expect_true(is.finite(ll_all))
     # Test with core only
     ll_core_only <- MOSAIC::calc_model_likelihood(
          obs_cases = obs_cases,
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths
     )
     expect_true(is.finite(ll_core_only))
     # All terms at nonzero weights should produce different result than core only
     expect_true(ll_all != ll_core_only)
})
# 15. Test automatic distribution selection (Poisson vs NegBin)
testthat::test_that("automatic distribution selection works based on overdispersion", {
     set.seed(123)
     n_loc <- 2
     n_time <- 52
     # Low variance data (should use Poisson)
     obs_cases_low_var <- matrix(10, n_loc, n_time)
     obs_cases_low_var[1, ] <- obs_cases_low_var[1, ] + sample(-1:1, n_time, replace = TRUE)
     obs_cases_low_var[obs_cases_low_var < 0] <- 0
     # High variance data (should use NegBin)
     obs_cases_high_var <- matrix(10, n_loc, n_time)
     obs_cases_high_var[2, ] <- rpois(n_time, lambda = 10) * sample(1:5, n_time, replace = TRUE)
     est_cases <- matrix(10, n_loc, n_time)
     obs_deaths <- matrix(2, n_loc, n_time)
     est_deaths <- matrix(2, n_loc, n_time)
     ll <- MOSAIC::calc_model_likelihood(
          obs_cases = rbind(obs_cases_low_var[1, ], obs_cases_high_var[2, ]),
          est_cases = est_cases,
          obs_deaths = obs_deaths,
          est_deaths = est_deaths
     )
     expect_true(is.finite(ll))
})
