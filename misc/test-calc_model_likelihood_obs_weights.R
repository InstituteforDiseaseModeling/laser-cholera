# ============================================================================
# Tests for per-observation (per-cell) confidence weighting in
# calc_model_likelihood() -- weights_obs_cases / weights_obs_deaths.
#
# All expected values are hand-computed against the NB log-density used by the
# core scorer (calc_log_likelihood_distributions.R):
#   ll_i = lgamma(o+k) - lgamma(k) - lgamma(o+1)
#          + k*log(k/(k+e)) + o*log(e/(k+e))         (e floored at 1e-10)
# and the per-location weighted sum ll = sum(w_eff * ll_vec).
#
# Design invariants under test:
#  (1) all-ones weights_obs == NULL -> byte-identical LL AND identical gate.
#  (2) a 0.5-weight cell carries exactly half the per-cell weight of a 1.0 cell.
#  (3) per-location effective-weight mass == masked-weights_time sum (B-1).
#  (4) a fully-zeroed weight row contributes 0.
#  (5) the exactly-3-observed-cell boundary still passes under all-ones.
#  (6) a documented-zero-heavy country does not dominate (magnitude bounded).
#  (7) dimension validation on the new matrix args.
# ============================================================================

# Hand NB log-density (mirrors the scorer's negbin kernel)
.nb_logdens <- function(o, e, k) {
     e <- max(e, 1e-10)
     lgamma(o + k) - lgamma(k) - lgamma(o + 1) +
          k * log(k / (k + e)) + o * log(e / (k + e))
}

# ---------------------------------------------------------------------------
# (1) all-ones weights_obs == NULL : byte-identical LL
# ---------------------------------------------------------------------------
testthat::test_that("all-ones weights_obs is byte-identical to NULL", {
     obs_c <- matrix(c(0, 5, 9, 2, 1, 4), nrow = 2, byrow = TRUE)
     est_c <- matrix(c(1, 4, 8, 3, 2, 5), nrow = 2, byrow = TRUE)
     obs_d <- matrix(c(0, 1, 2, 0, 0, 1), nrow = 2, byrow = TRUE)
     est_d <- matrix(c(1, 1, 1, 1, 1, 1), nrow = 2, byrow = TRUE)

     ll_null <- MOSAIC::calc_model_likelihood(obs_c, est_c, obs_d, est_d)
     ll_ones <- MOSAIC::calc_model_likelihood(
          obs_c, est_c, obs_d, est_d,
          weights_obs_cases  = matrix(1, 2, 3),
          weights_obs_deaths = matrix(1, 2, 3)
     )
     # Byte identity is the acceptance gate: identical(), not just ==.
     expect_identical(ll_null, ll_ones)
})

testthat::test_that("all-ones weights_obs yields identical gate decision", {
     # Location with exactly 3 finite obs (boundary) must pass under all-ones
     # exactly as under NULL -- i.e. the location still contributes (not 0/NA).
     obs_c <- matrix(c(0, 5, 9, NA), nrow = 1)
     est_c <- matrix(c(1, 4, 8, 3),  nrow = 1)
     zd    <- matrix(0, 1, 4)

     ll_null <- MOSAIC::calc_model_likelihood(obs_c, est_c, zd, zd)
     ll_ones <- MOSAIC::calc_model_likelihood(
          obs_c, est_c, zd, zd, weights_obs_cases = matrix(1, 1, 4)
     )
     expect_identical(ll_null, ll_ones)
     expect_true(is.finite(ll_null))
     # The cases block contributed (3 finite obs passes the >=3 gate) -> nonzero.
     expect_true(ll_null != 0)
})

# ---------------------------------------------------------------------------
# (2) a 0.5-weight cell carries exactly half the per-cell weight of a 1.0 cell
# ---------------------------------------------------------------------------
testthat::test_that("a 0.5-weight cell carries exactly half a 1.0 cell's weight", {
     obs_c <- matrix(c(0, 5, 9, 2), nrow = 1)
     est_c <- matrix(c(1, 4, 8, 3), nrow = 1)
     wt    <- rep(1, 4)
     wobs  <- c(1, 0.5, 1, 1)

     w_eff <- MOSAIC:::.weights_obs_effective(wt, wobs, obs_c[1, ], est_c[1, ])
     # Cell 2 (w=0.5) must be exactly half of any w=1 cell after renorm.
     expect_equal(w_eff[2] / w_eff[1], 0.5, tolerance = 1e-12)
     expect_equal(w_eff[2] / w_eff[3], 0.5, tolerance = 1e-12)
     expect_equal(w_eff[2] / w_eff[4], 0.5, tolerance = 1e-12)
})

testthat::test_that("weighted NB LL equals sum(w_eff * NB log-density) by hand", {
     obs_c <- matrix(c(0, 5, 9, 2), nrow = 1)
     est_c <- matrix(c(1, 4, 8, 3), nrow = 1)
     zd    <- matrix(0, 1, 4)
     wobs  <- matrix(c(1, 0.5, 1, 1), nrow = 1)

     # Reconstruct the effective weight + k exactly as the scorer does.
     w_eff <- MOSAIC:::.weights_obs_effective(rep(1, 4), wobs[1, ], obs_c[1, ], est_c[1, ])
     k     <- MOSAIC:::.nb_size_from_obs_weighted(obs_c[1, ], w_eff, k_min = 3)
     ll_vec   <- vapply(1:4, function(i) .nb_logdens(obs_c[1, i], est_c[1, i], k), numeric(1))
     expected <- sum(w_eff * ll_vec)   # deaths all-zero-data contribute 0

     ll <- MOSAIC::calc_model_likelihood(obs_c, est_c, zd, zd, weights_obs_cases = wobs)
     expect_equal(ll, expected, tolerance = 1e-10)
})

# ---------------------------------------------------------------------------
# (3) per-location effective-weight mass == masked-weights_time sum
# ---------------------------------------------------------------------------
testthat::test_that("effective-weight mass equals masked-weights_time sum", {
     obs   <- c(0, 5, NA, 9, 2)       # one NA cell -> masked out
     est   <- c(1, 4, 3,  8, 3)
     wt    <- c(2, 1, 1, 1, 0.5)      # non-uniform time weights
     wobs  <- c(0.8, 0.9, 0.8, 0.95, 0.8)

     target <- sum(MOSAIC:::.mask_weights(wt, obs, est))  # masked weights_time sum
     w_eff  <- MOSAIC:::.weights_obs_effective(wt, wobs, obs, est)
     expect_equal(sum(w_eff), target, tolerance = 1e-12)
     # Masked cell must stay zero.
     expect_equal(w_eff[3], 0, tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# (4) a fully-zeroed weight row contributes 0
# ---------------------------------------------------------------------------
testthat::test_that("a fully-zeroed weight row contributes zero LL", {
     obs_c <- matrix(c(0, 5, 9, 2), nrow = 1)
     est_c <- matrix(c(1, 4, 8, 3), nrow = 1)
     zd    <- matrix(0, 1, 4)         # deaths perfect-match zero-data -> 0

     ll <- MOSAIC::calc_model_likelihood(
          obs_c, est_c, zd, zd, weights_obs_cases = matrix(0, 1, 4)
     )
     expect_equal(ll, 0, tolerance = 1e-12)

     # Helper returns all zeros for a zeroed row.
     w_eff <- MOSAIC:::.weights_obs_effective(rep(1, 4), rep(0, 4), obs_c[1, ], est_c[1, ])
     expect_true(all(w_eff == 0))
})

# ---------------------------------------------------------------------------
# (5) exactly-3-observed-cell boundary passes under all-ones
# ---------------------------------------------------------------------------
testthat::test_that("exactly-3-observed boundary passes under all-ones (gate back-compat)", {
     obs_c <- matrix(c(0, 5, 9, NA), nrow = 1)   # 3 finite -> passes >=3
     est_c <- matrix(c(1, 4, 8, 3),  nrow = 1)
     zd    <- matrix(0, 1, 4)

     ll_ones <- MOSAIC::calc_model_likelihood(
          obs_c, est_c, zd, zd, weights_obs_cases = matrix(1, 1, 4)
     )
     expect_true(is.finite(ll_ones))
     expect_true(ll_ones != 0)   # cases block contributed

     # 2 finite obs -> fails the >=3 gate -> cases contributes 0; deaths zero -> total 0.
     obs_c2 <- matrix(c(0, 5, NA, NA), nrow = 1)
     ll2 <- MOSAIC::calc_model_likelihood(
          obs_c2, est_c, zd, zd, weights_obs_cases = matrix(1, 1, 4)
     )
     expect_equal(ll2, 0, tolerance = 1e-12)
})

# ---------------------------------------------------------------------------
# (6) documented-zero-heavy country does not dominate (magnitude bounded)
# ---------------------------------------------------------------------------
testthat::test_that("documented-zero-heavy country LL stays bounded vs unweighted", {
     T <- 50
     obs <- matrix(0, 1, T); obs[1, 1:5] <- c(3, 8, 12, 6, 2)  # 5 real, 45 zeros
     est <- matrix(2, 1, T); est[1, 1:5] <- c(2, 7, 10, 5, 3)
     zd  <- matrix(0, 1, T)

     ll_unw <- MOSAIC::calc_model_likelihood(obs, est, zd, zd)
     wobs   <- matrix(0.80, 1, T); wobs[1, 1:5] <- 0.95         # documented_zero=0.80
     ll_w   <- MOSAIC::calc_model_likelihood(obs, est, zd, zd, weights_obs_cases = wobs)

     expect_true(is.finite(ll_w))
     # Mass-preservation keeps the weighted LL the same order of magnitude as the
     # unweighted one -- it must not explode (Lesson #5 class) nor collapse to ~0.
     expect_lt(abs(ll_w), abs(ll_unw) * 1.05)
     expect_gt(abs(ll_w), abs(ll_unw) * 0.5)
})

# ---------------------------------------------------------------------------
# (7) dimension validation
# ---------------------------------------------------------------------------
testthat::test_that("weights_obs dimension mismatch errors", {
     obs <- matrix(0, 2, 3); est <- matrix(0, 2, 3)
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs, est, obs, est, weights_obs_cases = matrix(1, 2, 4)
          ),
          "weights_obs_cases must have the same dimensions"
     )
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs, est, obs, est, weights_obs_deaths = matrix(1, 1, 3)
          ),
          "weights_obs_deaths must have the same dimensions"
     )
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs, est, obs, est, weights_obs_cases = as.vector(matrix(1, 2, 3))
          ),
          "weights_obs_cases must be a matrix"
     )
     expect_error(
          MOSAIC::calc_model_likelihood(
               obs, est, obs, est, weights_obs_cases = matrix(-0.1, 2, 3)
          ),
          "weights_obs_cases must be >= 0"
     )
})

# ---------------------------------------------------------------------------
# Independent gates for cases vs deaths
# ---------------------------------------------------------------------------
testthat::test_that("cases and deaths gates are independent under weighting", {
     # Cases: 4 observed at 0.95 -> ESS 3.8 >= 3 passes.
     # Deaths: 4 observed but all at 0.5 -> ESS 2.0 < 3 fails -> deaths contributes 0.
     obs_c <- matrix(c(3, 8, 5, 2), nrow = 1)
     est_c <- matrix(c(2, 7, 6, 3), nrow = 1)
     obs_d <- matrix(c(1, 2, 1, 0), nrow = 1)
     est_d <- matrix(c(1, 1, 1, 1), nrow = 1)
     wc    <- matrix(0.95, 1, 4)
     wd    <- matrix(0.50, 1, 4)

     # Compare to a run where deaths is forced off via a zeroed gate, to confirm
     # the deaths block dropped out (cases-only LL).
     ll_both <- MOSAIC::calc_model_likelihood(
          obs_c, est_c, obs_d, est_d, weights_obs_cases = wc, weights_obs_deaths = wd
     )
     ll_cases_only <- MOSAIC::calc_model_likelihood(
          obs_c, est_c, obs_d, est_d, weights_obs_cases = wc,
          weights_obs_deaths = matrix(0, 1, 4)
     )
     # Deaths ESS (2.0) < 3 so deaths dropped in both -> identical.
     expect_equal(ll_both, ll_cases_only, tolerance = 1e-12)
})
