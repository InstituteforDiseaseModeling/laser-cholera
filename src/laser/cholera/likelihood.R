###############################################################################
## calc_model_likelihood.R  (Core NB + optional shape terms; no guardrails)
###############################################################################

#' Compute the total model likelihood
#'
#' Scores model fits against observed data using a Negative Binomial (NB)
#' time-series log-likelihood per location and outcome (cases, deaths) with
#' a weighted MoM dispersion estimate and a \code{k_min} floor.
#'
#' Optional shape terms are enabled by setting their weight > 0: peak timing
#' (Normal), peak magnitude (log-Normal with adaptive sigma), cumulative
#' progression (NB at cumulative fractions), and Weighted Interval Score (WIS).
#' All weights default to 0 (OFF).
#'
#' Shape terms are internally T-normalized so that weight parameters share
#' a common scale: \code{weight = 0.25} means the term contributes roughly 25
#' percent as much as the NB core. Peaks are scaled by \code{T / N_peaks},
#' cumulative and WIS by \code{T} (both return per-evaluation averages).
#'
#' Non-finite per-location LL values are replaced with \code{-Inf} (zero
#' importance weight). The NB likelihood naturally produces very negative
#' scores for bad fits without needing artificial guardrails.
#'
#' @param obs_cases,est_cases Matrices \code{n_locations x n_time_steps}.
#' @param obs_deaths,est_deaths Matrices \code{n_locations x n_time_steps}.
#' @param weight_cases,weight_deaths Scalar weights for case/death blocks. Default 1.
#' @param weights_location Length-\code{n_locations} non-negative weights.
#' @param weights_time Length-\code{n_time_steps} non-negative weights.
#' @param config Optional LASER config list (location_name, date_start, date_stop).
#' @param nb_k_min_cases Minimum NB dispersion floor for cases. Default \code{3}.
#' @param nb_k_min_deaths Minimum NB dispersion floor for deaths. Default \code{3}.
#' @param verbose If \code{TRUE}, prints component summaries per location.
#' @param weight_peak_timing,weight_peak_magnitude Weights for peak terms
#'   (T-normalized). Default \code{0} (OFF). Set > 0 to enable; 0.25 = 25 percent
#'   of NB core influence.
#' @param weight_cumulative_total Weight for cumulative progression (T-normalized).
#'   Default \code{0} (OFF). Cumulative helper is /end_idx normalized so weights
#'   are on the same scale as other shape terms.
#' @param weight_wis Weight for WIS term (T-normalized). Default \code{0} (OFF).
#'   Ablation tests show 0.10 provides trajectory-shape regularization.
#' @param sigma_peak_time SD (weeks) for peak timing Normal; default \code{1}.
#' @param sigma_peak_log Base SD on log-scale for peak magnitude; default \code{0.5}.
#' @param wis_quantiles Quantiles for WIS if enabled.
#' @param cumulative_timepoints Fractions for cumulative progression.
#'
#' @return Scalar total log-likelihood (finite), \code{-Inf} if non-finite,
#'   or \code{NA_real_} if all locations contribute nothing.
#' @export
calc_model_likelihood <- function(obs_cases,
                                  est_cases,
                                  obs_deaths,
                                  est_deaths,
                                  weight_cases     = NULL,
                                  weight_deaths    = NULL,
                                  weights_location = NULL,
                                  weights_time     = NULL,
                                  config           = NULL,
                                  nb_k_min_cases   = 3,
                                  nb_k_min_deaths  = 3,
                                  verbose          = FALSE,
                                  # ---- shape term weights (0 = OFF; 0.25 = 25% of NB core) ----
                                  weight_peak_timing       = 0,
                                  weight_peak_magnitude    = 0,
                                  weight_cumulative_total  = 0,
                                  weight_wis               = 0,
                                  # ---- peak controls ----
                                  sigma_peak_time  = 1,
                                  sigma_peak_log   = 0.5,
                                  # ---- WIS (optional) ----
                                  wis_quantiles      = c(0.025, 0.25, 0.5, 0.75, 0.975),
                                  # ---- cumulative progression ----
                                  cumulative_timepoints = c(0.25, 0.5, 0.75, 1.0))
{
     # --- basic checks ---
     if (!is.matrix(obs_cases) || !is.matrix(est_cases) ||
         !is.matrix(obs_deaths) || !is.matrix(est_deaths)) {
          stop("all inputs must be matrices.")
     }

     # Validation: Check for negative estimated values
     if (any(est_cases < 0, na.rm = TRUE) || any(est_deaths < 0, na.rm = TRUE)) {
          stop("Estimated values must be non-negative.")
     }

     n_locations  <- nrow(obs_cases)
     n_time_steps <- ncol(obs_cases)

     if (any(dim(est_cases)   != c(n_locations, n_time_steps)) ||
         any(dim(obs_deaths)  != c(n_locations, n_time_steps)) ||
         any(dim(est_deaths)  != c(n_locations, n_time_steps))) {
          stop("All matrices must have the same dimensions (n_locations x n_time_steps).")
     }

     if (is.null(weights_location)) weights_location <- rep(1, n_locations)
     if (is.null(weights_time))     weights_time     <- rep(1, n_time_steps)
     if (is.null(weight_cases))     weight_cases     <- 1
     if (is.null(weight_deaths))    weight_deaths    <- 1

     if (length(weights_location) != n_locations) stop("weights_location must match n_locations.")
     if (length(weights_time)     != n_time_steps) stop("weights_time must match n_time_steps.")
     if (any(weights_location < 0) || any(weights_time < 0)) stop("All weights must be >= 0.")
     if (sum(weights_location) == 0 || sum(weights_time) == 0) stop("weights_location and weights_time must not all be zero.")

     # --- precompute peak indices per location (once, not per call) ---
     peak_indices_by_loc <- NULL
     timestep_to_weeks <- 7  # default: daily timesteps, divide by 7 to get weeks
     if ((weight_peak_timing > 0 || weight_peak_magnitude > 0) && !is.null(config)) {
          location_names <- config$location_name
          date_start_cfg <- config$date_start
          date_stop_cfg <- config$date_stop

          if (!is.null(location_names) && !is.null(date_start_cfg) && !is.null(date_stop_cfg)) {
               # Build date sequence once; detect timestep resolution
               date_seq <- seq(as.Date(date_start_cfg), as.Date(date_stop_cfg), by = "day")
               if (length(date_seq) != n_time_steps) {
                    date_seq <- seq(as.Date(date_start_cfg), as.Date(date_stop_cfg), by = "week")
                    if (length(date_seq) != n_time_steps) {
                         date_seq <- NULL
                    } else {
                         timestep_to_weeks <- 1  # weekly data: 1 timestep = 1 week
                    }
               }

               if (!is.null(date_seq)) {
                    epidemic_peaks <- MOSAIC::epidemic_peaks
                    peak_indices_by_loc <- vector("list", n_locations)
                    for (j_pk in seq_len(n_locations)) {
                         iso_code <- if (j_pk <= length(location_names)) location_names[j_pk] else NA_character_
                         if (is.na(iso_code)) { peak_indices_by_loc[[j_pk]] <- integer(0); next }
                         loc_peaks <- epidemic_peaks[epidemic_peaks$iso_code == iso_code, ]
                         if (nrow(loc_peaks) == 0) { peak_indices_by_loc[[j_pk]] <- integer(0); next }
                         idx <- vapply(loc_peaks$peak_date, function(pd) {
                              which.min(abs(date_seq - as.Date(pd)))
                         }, integer(1))
                         peak_indices_by_loc[[j_pk]] <- idx[idx > 0L & idx <= n_time_steps]
                    }
               }
          }
     }

     # --- main loop ---
     ll_locations <- rep(NA_real_, n_locations)

     for (j in seq_len(n_locations)) {

          obs_c <- obs_cases[j, ]; est_c <- est_cases[j, ]
          obs_d <- obs_deaths[j, ]; est_d <- est_deaths[j, ]

          # Require minimum observations for meaningful likelihood
          min_obs_for_likelihood <- 3
          have_cases  <- sum(is.finite(obs_c)) >= min_obs_for_likelihood
          have_deaths <- sum(is.finite(obs_d)) >= min_obs_for_likelihood

          # Weighted NB dispersion (k) estimated from observed data via method-of-moments.
          # Note: k is a property of the observation process, not the model-observation
          # mismatch, so all simulations are evaluated against the same k. This is
          # intentional — it ensures the likelihood reflects data noise characteristics
          # rather than calibration quality. The k_min floor prevents the NB from
          # collapsing to a near-Poisson kernel for low-variance series.
          k_c <- if (have_cases)  nb_size_from_obs_weighted(obs_c, weights_time, k_min = nb_k_min_cases) else Inf
          k_d <- if (have_deaths) nb_size_from_obs_weighted(obs_d, weights_time, k_min = nb_k_min_deaths) else Inf

          # Core NB time series LL (pass k and k_min explicitly)
          ll_cases  <- if (have_cases) MOSAIC::calc_log_likelihood(
               observed  = obs_c,
               estimated = est_c,
               family    = "negbin",
               weights   = mask_weights(weights_time, obs_c, est_c),
               k         = k_c,
               k_min     = nb_k_min_cases,
               verbose   = FALSE
          ) else 0

          ll_deaths <- if (have_deaths) MOSAIC::calc_log_likelihood(
               observed  = obs_d,
               estimated = est_d,
               family    = "negbin",
               weights   = mask_weights(weights_time, obs_d, est_d),
               k         = k_d,
               k_min     = nb_k_min_deaths,
               verbose   = FALSE
          ) else 0

          # Peak-based likelihoods using precomputed peak indices
          ll_peak_time_c <- ll_peak_time_d <- 0
          ll_peak_mag_c <- ll_peak_mag_d <- 0

          if ((weight_peak_timing > 0 || weight_peak_magnitude > 0) && !is.null(peak_indices_by_loc)) {
               loc_peak_idx <- peak_indices_by_loc[[j]]
               if (length(loc_peak_idx) > 0) {
                    if (weight_peak_timing > 0) {
                         if (have_cases) {
                              ll_peak_time_c <- .calc_peak_timing_from_indices(
                                   est_c, loc_peak_idx, sigma_peak_time,
                                   timestep_to_weeks = timestep_to_weeks
                              )
                         }
                         if (have_deaths) {
                              ll_peak_time_d <- .calc_peak_timing_from_indices(
                                   est_d, loc_peak_idx, sigma_peak_time,
                                   timestep_to_weeks = timestep_to_weeks
                              )
                         }
                    }
                    if (weight_peak_magnitude > 0) {
                         if (have_cases) {
                              ll_peak_mag_c <- .calc_peak_magnitude_from_indices(
                                   obs_c, est_c, loc_peak_idx, sigma_peak_log
                              )
                         }
                         if (have_deaths) {
                              ll_peak_mag_d <- .calc_peak_magnitude_from_indices(
                                   obs_d, est_d, loc_peak_idx, sigma_peak_log
                              )
                         }
                    }
               }
          }

          # Cumulative progression (using data-driven k)
          ll_cum_tot_c <- ll_cum_tot_d <- 0
          if (weight_cumulative_total > 0) {
               if (have_cases)  ll_cum_tot_c <- ll_cumulative_progressive_nb(obs_c, est_c, cumulative_timepoints, k_c, weights_time)
               if (have_deaths) ll_cum_tot_d <- ll_cumulative_progressive_nb(obs_d, est_d, cumulative_timepoints, k_d, weights_time)
          }


          # WIS (optional) — raw negated WIS; weight_wis applied at assembly (like other components)
          ll_wis_cases <- ll_wis_deaths <- 0
          if (weight_wis > 0) {
               if (have_cases) {
                    wis_c <- compute_wis_parametric_row(obs_c, est_c, weights_time, wis_quantiles, k_use = k_c)
                    if (is.finite(wis_c)) ll_wis_cases <- -wis_c
               }
               if (have_deaths) {
                    wis_d <- compute_wis_parametric_row(obs_d, est_d, weights_time, wis_quantiles, k_use = k_d)
                    if (is.finite(wis_d)) ll_wis_deaths <- -wis_d
               }
          }


          # Assembly formula (per location j):
          #
          # Shape term scaling: N_obs / N_component_observations
          #
          # Each shape term helper returns O(1) (per-observation scale). The scale
          # factor inflates it to O(N_obs) to match the NB core. The denominator
          # is the number of independent observations for that component:
          #
          #   NB core:     N_obs observations → no scaling (reference)
          #   Peaks:       N_peaks observations → scale by N_obs / N_peaks
          #   WIS:         N_quantiles evaluations → scale by N_obs / N_quantiles
          #   Cumulative:  N_eval_points evaluations → scale by N_obs / N_eval_points
          #
          # This makes w=0.05 mean "~5% of NB core influence" for ALL shape terms.
          #
          #   ll_loc = wc * NB_cases + wd * NB_deaths
          #     + (N_obs/N_peaks)      * w_pt  * (wc * pt_c  + wd * pt_d)
          #     + (N_obs/N_peaks)      * w_pm  * (wc * pm_c  + wd * pm_d)
          #     + (N_obs/N_eval_pts)   * w_cum * (wc * cum_c + wd * cum_d)
          #     + (N_obs/N_quantiles)  * w_wis * (wc * wis_c + wd * wis_d)
          #
          # NOTE: weight_cases/weight_deaths apply multiplicatively to EVERY component.

          # N_obs: count of timesteps with at least one finite observation
          N_obs <- sum(is.finite(obs_c) | is.finite(obs_d))

          # Component observation counts
          n_peaks_j     <- if (!is.null(peak_indices_by_loc)) length(peak_indices_by_loc[[j]]) else 0L
          n_wis_quant   <- length(wis_quantiles)
          n_cum_points  <- length(cumulative_timepoints)

          # Scale factors: N_obs / N_component_obs
          peak_scale <- if (n_peaks_j > 0) N_obs / n_peaks_j else 0
          wis_scale  <- if (n_wis_quant > 0) N_obs / n_wis_quant else 0
          cum_scale  <- if (n_cum_points > 0) N_obs / n_cum_points else 0

          ll_loc_core <-
               weight_cases  * ll_cases +
               weight_deaths * ll_deaths

          ll_loc_peaks <-
               peak_scale * weight_peak_timing    * (weight_cases * ll_peak_time_c + weight_deaths * ll_peak_time_d) +
               peak_scale * weight_peak_magnitude * (weight_cases * ll_peak_mag_c  + weight_deaths * ll_peak_mag_d)

          ll_loc_cum <-
               cum_scale * weight_cumulative_total * (weight_cases * ll_cum_tot_c + weight_deaths * ll_cum_tot_d)

          ll_loc_wis <-
               wis_scale * weight_wis * (weight_cases * ll_wis_cases + weight_deaths * ll_wis_deaths)

          ll_loc_total <- ll_loc_core + ll_loc_peaks + ll_loc_cum + ll_loc_wis

          # Non-finite safety net: -Inf gets zero importance weight
          if (!is.finite(ll_loc_total)) {
               ll_locations[j] <- -Inf
               next
          }

          ll_locations[j] <- weights_location[j] * ll_loc_total

          if (verbose) {
               message(sprintf(
                    "Location %d: core=%.2f | peaks=%.2f | cum=%.2f | wis=%.2f -> weighted=%.2f",
                    j, ll_loc_core, ll_loc_peaks, ll_loc_cum, ll_loc_wis,
                    weights_location[j] * ll_loc_total
               ))
          }
     }

     if (all(is.na(ll_locations))) {
          if (verbose) message("All locations contributed NA — returning NA.")
          return(NA_real_)
     }

     ll_total <- sum(ll_locations, na.rm = TRUE)
     if (!is.finite(ll_total)) ll_total <- -Inf
     if (verbose) message(sprintf("Overall total log-likelihood: %.2f", ll_total))
     ll_total
}

###############################################################################
## Helpers (ALL defined outside the main function)
###############################################################################

# Mask weights on non-finite entries
mask_weights <- function(w, obs_vec, est_vec = NULL) {
     w2 <- w
     bad <- !is.finite(obs_vec) | (!is.null(est_vec) & !is.finite(est_vec))
     if (any(bad)) w2[bad] <- 0
     w2
}



# --- Fast peak helpers using precomputed indices (no date parsing) ---

# Peak timing likelihood from precomputed indices
.calc_peak_timing_from_indices <- function(est_vec, peak_indices, sigma_peak_time = 1,
                                           timestep_to_weeks = 7) {
     ll_total <- 0
     n_ts <- length(est_vec)
     for (peak_idx in peak_indices) {
          window <- max(1L, peak_idx - 14L):min(n_ts, peak_idx + 14L)
          if (length(window) > 2L) {
               est_peak_idx <- window[which.max(est_vec[window])]
               time_diff <- (est_peak_idx - peak_idx) / timestep_to_weeks
               ll_total <- ll_total + stats::dnorm(time_diff, 0, sigma_peak_time, log = TRUE)
          }
     }
     ll_total
}

# Peak magnitude likelihood from precomputed indices
.calc_peak_magnitude_from_indices <- function(obs_vec, est_vec, peak_indices,
                                              sigma_peak_log = 0.5) {
     ll_total <- 0
     n_ts <- length(obs_vec)
     for (peak_idx in peak_indices) {
          window <- max(1L, peak_idx - 14L):min(n_ts, peak_idx + 14L)
          if (length(window) > 2L) {
               obs_peak_val <- max(obs_vec[window], na.rm = TRUE)
               est_peak_val <- max(est_vec[window], na.rm = TRUE)
               if (is.finite(obs_peak_val) && is.finite(est_peak_val) &&
                   obs_peak_val > 0 && est_peak_val > 0) {
                    adaptive_sigma <- sigma_peak_log * sqrt(100 / max(obs_peak_val, 100))
                    ll_total <- ll_total + stats::dnorm(
                         log(est_peak_val) - log(obs_peak_val), 0, adaptive_sigma, log = TRUE
                    )
               }
          }
     }
     ll_total
}

# --- Legacy peak helpers (retained for external/standalone use) ---

# Peak timing likelihood using epidemic_peaks data
calc_multi_peak_timing_ll <- function(obs_vec, est_vec, iso_code = NULL,
                                     date_start = NULL, date_stop = NULL,
                                     sigma_peak_time = 1) {
     # Use lazy-loaded package data (cached in namespace after first access)
     epidemic_peaks <- MOSAIC::epidemic_peaks

     # If required info missing, return 0
     if (is.null(iso_code) || is.null(date_start) || is.null(date_stop)) return(0)

     # Get peaks for this location
     loc_peaks <- epidemic_peaks[epidemic_peaks$iso_code == iso_code, ]
     if (nrow(loc_peaks) == 0) return(0)

     # Create date sequence for the time series
     date_seq <- seq(as.Date(date_start), as.Date(date_stop), by = "day")
     if (length(date_seq) != length(obs_vec)) {
          # Try weekly if daily doesn't match
          date_seq <- seq(as.Date(date_start), as.Date(date_stop), by = "week")
          if (length(date_seq) != length(obs_vec)) return(0)  # Can't match dates
     }

     # Convert peak dates to indices
     peak_indices <- numeric(nrow(loc_peaks))
     for (i in 1:nrow(loc_peaks)) {
          idx <- which.min(abs(date_seq - as.Date(loc_peaks$peak_date[i])))
          if (length(idx) > 0) peak_indices[i] <- idx[1]
     }
     peak_indices <- peak_indices[peak_indices > 0 & peak_indices <= length(obs_vec)]

     if (length(peak_indices) == 0) return(0)

     # Find peaks in estimated data near the expected peak times
     ll_total <- 0
     n_matched <- 0

     for (peak_idx in peak_indices) {
          # Look for peak in estimated data within window
          window <- max(1, peak_idx - 14):min(length(est_vec), peak_idx + 14)  # +/- 2 weeks
          if (length(window) > 2) {
               est_peak_idx <- window[which.max(est_vec[window])]
               # Calculate timing difference in weeks
               time_diff <- (est_peak_idx - peak_idx) / 7
               # Calculate log-likelihood for this peak timing
               ll_total <- ll_total + stats::dnorm(time_diff, 0, sigma_peak_time, log = TRUE)
               n_matched <- n_matched + 1
          }
     }

     return(ll_total)
}

# Peak magnitude likelihood using epidemic_peaks data
calc_multi_peak_magnitude_ll <- function(obs_vec, est_vec, iso_code = NULL,
                                        date_start = NULL, date_stop = NULL,
                                        sigma_peak_log = 0.5) {
     # Use lazy-loaded package data (cached in namespace after first access)
     epidemic_peaks <- MOSAIC::epidemic_peaks

     # If required info missing, return 0
     if (is.null(iso_code) || is.null(date_start) || is.null(date_stop)) return(0)

     # Get peaks for this location
     loc_peaks <- epidemic_peaks[epidemic_peaks$iso_code == iso_code, ]
     if (nrow(loc_peaks) == 0) return(0)
     
     # Create date sequence for the time series
     date_seq <- seq(as.Date(date_start), as.Date(date_stop), by = "day")
     if (length(date_seq) != length(obs_vec)) {
          # Try weekly if daily doesn't match
          date_seq <- seq(as.Date(date_start), as.Date(date_stop), by = "week")
          if (length(date_seq) != length(obs_vec)) return(0)  # Can't match dates
     }
     
     # Convert peak dates to indices
     peak_indices <- numeric(nrow(loc_peaks))
     for (i in 1:nrow(loc_peaks)) {
          idx <- which.min(abs(date_seq - as.Date(loc_peaks$peak_date[i])))
          if (length(idx) > 0) peak_indices[i] <- idx[1]
     }
     peak_indices <- peak_indices[peak_indices > 0 & peak_indices <= length(obs_vec)]
     
     if (length(peak_indices) == 0) return(0)
     
     # Calculate magnitude likelihood for peaks
     ll_total <- 0
     n_matched <- 0
     
     for (peak_idx in peak_indices) {
          # Get observed magnitude at expected peak time
          window <- max(1, peak_idx - 14):min(length(obs_vec), peak_idx + 14)  # +/- 2 weeks
          if (length(window) > 2) {
               obs_peak_val <- max(obs_vec[window], na.rm = TRUE)
               est_peak_val <- max(est_vec[window], na.rm = TRUE)
               
               if (is.finite(obs_peak_val) && is.finite(est_peak_val) && 
                   obs_peak_val > 0 && est_peak_val > 0) {
                    # Adaptive sigma that scales with peak size
                    adaptive_sigma <- sigma_peak_log * sqrt(100 / max(obs_peak_val, 100))
                    # Log-normal likelihood on the ratio
                    ll_mag <- stats::dnorm(log(est_peak_val) - log(obs_peak_val), 
                                         0, adaptive_sigma, log = TRUE)
                    ll_total <- ll_total + ll_mag
                    n_matched <- n_matched + 1
               }
          }
     }
     
     return(ll_total)
}

# Robust cumulative NB progression
ll_cumulative_progressive_nb <- function(obs_vec,
                                         est_vec,
                                         timepoints = c(0.25, 0.5, 0.75, 1.0),
                                         k_data = NULL,
                                         weights_time = NULL) {
     n <- length(obs_vec)

     # Use weights if provided
     if (is.null(weights_time)) weights_time <- rep(1, n)

     # Fallback k when data-driven estimate is unavailable
     k_fallback <- getOption("MOSAIC.cumulative_k", 10)

     vals <- vector("numeric", length(timepoints))
     n_vals <- 0L

     for (tp in timepoints) {
          # Ensure index is at least 1 and at most n
          end_idx <- min(n, max(1L, round(n * tp)))

          # Scale k proportionally to the number of summed timesteps.
          # For independent NB(mu_i, k) variables, the sum's dispersion ≈ k * n_summed
          # (exact for identical means; reasonable upper bound for varying means).
          cum_k <- if (!is.null(k_data) && is.finite(k_data)) {
               k_data * end_idx
          } else {
               k_fallback
          }

          # Use plain cumulative sums (NA-safe).
          idx_range <- seq_len(end_idx)
          o_cum <- sum(obs_vec[idx_range], na.rm = TRUE)
          e_cum <- sum(est_vec[idx_range], na.rm = TRUE)

          if (!is.finite(o_cum) || !is.finite(e_cum)) next

          # Handle zero prediction the same way as the core NB:
          # proportional penalty for zero est with nonzero obs
          if (e_cum <= 0 && o_cum > 0) {
               n_vals <- n_vals + 1L
               vals[n_vals] <- (-round(o_cum) * log(1e6)) / end_idx
               next
          }
          e_cum <- if (e_cum <= 0) 1e-10 else e_cum

          ll_tp <- stats::dnbinom(round(o_cum), mu = e_cum, size = cum_k, log = TRUE)
          if (!is.finite(ll_tp)) {
               # Non-finite NB result: use proportional penalty as fallback
               n_vals <- n_vals + 1L
               vals[n_vals] <- (-round(o_cum) * log(1e6)) / end_idx
               next
          }
          # Normalize by end_idx to convert from "LL of sum of end_idx obs"
          # to "per-observation LL". This makes the helper output O(1),
          # allowing uniform T-scaling at assembly like other shape terms.
          n_vals <- n_vals + 1L
          vals[n_vals] <- ll_tp / end_idx
     }
     if (n_vals == 0L) return(0)  # No valid timepoints: contribute nothing
     mean(vals[seq_len(n_vals)])
}


# WIS helper (uses fixed k from core, or Poisson if Inf)
compute_wis_parametric_row <- function(y, est, w_time, probs, k_use) {
     # Early return for all-NA cases
     if (all(!is.finite(y)) || all(!is.finite(est))) return(NA_real_)
     
     w_use <- w_time
     bad <- !is.finite(y) | !is.finite(est)
     if (any(bad)) w_use[bad] <- 0
     if (sum(w_use) == 0) return(NA_real_)
     
     est_eval <- pmax(est, 1e-12)
     
     # Vectorized quantile functions
     qfun <- if (is.infinite(k_use)) {
          function(p) stats::qpois(p, lambda = est_eval)
     } else {
          function(p) stats::qnbinom(p, mu = est_eval, size = k_use)
     }
     
     probs  <- sort(unique(probs))
     has_med <- any(abs(probs - 0.5) < 1e-8)
     mae_term <- 0
     if (has_med) {
          # Fix: qfun returns a vector, need element-wise operations
          q_med <- qfun(0.5)
          mae_term <- 0.5 * sum(abs(y - q_med) * w_use, na.rm = TRUE) / sum(w_use)
     }
     
     lowers <- probs[probs < 0.5]
     uppers <- probs[probs > 0.5]
     pairs <- lapply(lowers, function(p) {
          complement <- 1 - p
          match_idx <- which(abs(uppers - complement) < 1e-8)
          c(p, if (length(match_idx) > 0) uppers[match_idx[1]] else uppers[which.min(abs(uppers - complement))])
     })
     K <- length(pairs)
     sum_IS <- 0
     
     if (K > 0) {
          for (pq in pairs) {
               pL <- pq[1]
               pU <- pq[2]
               # Fix: These return vectors, need element-wise operations
               qL <- qfun(pL)
               qU <- qfun(pU)
               alpha <- 1 - (pU - pL)
               # Vectorized operations
               width <- qU - qL
               under <- pmax(0, qL - y) * (2/alpha)
               over  <- pmax(0, y - qU) * (2/alpha)
               IS    <- width + under + over
               contrib <- sum(IS * w_use, na.rm = TRUE) / sum(w_use)
               sum_IS  <- sum_IS + (alpha/2) * contrib
          }
     }
     denom <- (K + 0.5)
     (mae_term + sum_IS) / denom
}

# Weighted method-of-moments NB dispersion (k) with floor.
# Uses Bessel-corrected weighted variance: V1^2 / (V1^2 - V2) normalisation,
# where V1 = sum(w) and V2 = sum(w^2). This avoids underestimating variance
# (and hence overestimating k) with small or unequal-weight samples.
#' @keywords internal
nb_size_from_obs_weighted <- function(x, w, k_min = 3, k_max = 1e5) {
     ok <- is.finite(x) & is.finite(w) & (w > 0)
     if (!any(ok)) return(Inf)
     x <- x[ok]; w <- w[ok]
     sw  <- sum(w)
     sw2 <- sum(w^2)
     m   <- sum(w * x) / sw
     # Bessel-corrected weighted variance: unbiased for frequency weights
     denom <- sw - sw2 / sw
     v <- if (denom > 0) sum(w * (x - m)^2) / denom else sum(w * (x - m)^2) / sw
     if (!is.finite(m) || !is.finite(v) || m <= 0 || v <= m) return(Inf)
     k  <- (m * m) / (v - m)
     k  <- max(min(k, k_max), k_min)
     k
}
