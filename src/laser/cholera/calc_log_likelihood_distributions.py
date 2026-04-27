"""Log-likelihood functions for Beta, Binomial, Gamma, NegBin, Normal, and Poisson.

Translated from calc_log_likelihood_distributions.R. Each function:
- Removes NaN/non-finite entries across observed, estimated, and weights before
  computing.
- Defaults all weights to 1 when not supplied.
- Logs diagnostics at INFO level when verbose=True.

R mapping notes:
    R `var(x)` / `sd(x)` use ddof=1; mapped to `np.var(x, ddof=1)` /
    `np.std(x, ddof=1)`.
    R `dbeta/dbinom/dgamma/dnorm/dpois(x, ..., log=TRUE)` map to
    `scipy.stats.*.(log)pmf/pdf`.
    R `dnbinom(x, size=k, mu=mu, log=TRUE)` maps to
    `scipy.stats.nbinom.logpmf(x, n=k, p=k/(k+mu))`.
    R `shapiro.test(x)` maps to `scipy.stats.shapiro(x)`.
    R `NA_real_` maps to `float("nan")`.
    R `message(...)` maps to `logger.info(...)`;
    R `warning(...)` maps to `logger.warning(...)`.
    R `is.na(x)` check uses `~np.isfinite(x)` (catches both NaN and Inf).
"""

import logging
from typing import Optional

import numpy as np
import scipy.stats

logger = logging.getLogger(__name__)


def calc_log_likelihood_beta(
    observed: np.ndarray,
    estimated: np.ndarray,
    mean_precision: bool = True,
    weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> float:
    """Calculate log-likelihood for Beta-distributed proportions.

    Computes the total log-likelihood for proportion data under the Beta
    distribution. Supports either the mean-precision parameterization (default)
    or the standard shape parameterization. Shape parameters are estimated from
    the data via method of moments.

    Args:
        observed: Observed values strictly in (0, 1).
        estimated: Model-predicted values strictly in (0, 1).
        mean_precision: If True (default), use mean-precision parameterization
            where phi = mu*(1-mu)/Var(residuals) - 1 and shape_1 = estimated*phi,
            shape_2 = (1-estimated)*phi. If False, estimate shape parameters
            directly from the observed vector.
        weights: Non-negative weights, same length as observed. Defaults to ones.
        verbose: If True, logs shape parameter estimates and total log-likelihood.

    Returns:
        Scalar log-likelihood. Returns float("nan") if all inputs are non-finite.

    Raises:
        ValueError: If lengths of observed and estimated do not match, any weights
            are negative, all weights are zero, observed or estimated values fall
            outside (0, 1), residual variance is non-positive, phi is non-positive,
            or estimated shape parameters are non-positive.

    Examples:
        >>> calc_log_likelihood_beta(
        ...     np.array([0.2, 0.6, 0.4]), np.array([0.25, 0.55, 0.35]),
        ...     verbose=False,
        ... )
    """
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(observed) != len(estimated):
        raise ValueError("Lengths of observed and estimated must match.")

    if weights is None:
        weights = np.ones(len(observed))
    else:
        weights = np.asarray(weights, dtype=float)

    # Remove NA triplets
    idx = np.where(np.isfinite(observed) & np.isfinite(estimated) & np.isfinite(weights))[0]
    observed = observed[idx]
    estimated = estimated[idx]
    weights = weights[idx]

    if len(observed) == 0 or len(estimated) == 0 or len(weights) == 0:
        if verbose:
            logger.info("No usable data (all NA) — returning NA for log-likelihood.")
        return float("nan")

    n = len(observed)
    if len(estimated) != n or len(weights) != n:
        raise ValueError("Lengths of observed, estimated, and weights must all match.")

    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero, cannot compute likelihood.")

    if np.any((observed <= 0) | (observed >= 1)):
        raise ValueError("observed must be strictly between 0 and 1 for Beta distribution.")
    if np.any((estimated <= 0) | (estimated >= 1)):
        raise ValueError("estimated must be strictly between 0 and 1 for Beta distribution.")

    if mean_precision:
        residuals = observed - estimated
        sigma2 = float(np.var(residuals, ddof=1))
        if sigma2 <= 0:
            raise ValueError("Residual variance is non-positive — cannot estimate phi.")
        mu = float(np.mean(observed))
        phi = (mu * (1 - mu)) / sigma2 - 1
        if phi <= 0:
            raise ValueError("Estimated phi must be > 0 — data may be too dispersed or flat.")
        shape_1 = estimated * phi
        shape_2 = (1 - estimated) * phi
        if verbose:
            logger.info("Mean–precision mode: estimated phi = %.2f", phi)
    else:
        mu = float(np.mean(observed))
        sigma2 = float(np.var(observed, ddof=1))
        shape_1_val = ((1 - mu) / sigma2 - 1 / mu) * mu**2
        shape_2_val = shape_1_val * (1 / mu - 1)
        if shape_1_val <= 0 or shape_2_val <= 0:
            raise ValueError("Estimated shape parameters must be positive — check observed values.")
        if verbose:
            logger.info(
                "Standard shape mode: shape_1 = %.2f, shape_2 = %.2f",
                shape_1_val,
                shape_2_val,
            )
        shape_1 = np.full(n, shape_1_val)
        shape_2 = np.full(n, shape_2_val)

    ll_vec = scipy.stats.beta.logpdf(observed, shape_1, shape_2)
    ll = float(np.sum(weights * ll_vec))

    if verbose:
        logger.info("Beta log-likelihood: %.2f", ll)
    return ll


def calc_log_likelihood_binomial(
    observed: np.ndarray,
    estimated: np.ndarray,
    trials: np.ndarray,
    weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> float:
    """Calculate log-likelihood for Binomial-distributed count data.

    Computes the total weighted log-likelihood for integer counts of successes
    under the Binomial distribution.

    Args:
        observed: Integer counts of successes (non-negative, <= trials).
        estimated: Expected success probabilities in (0, 1), same length as
            observed.
        trials: Total trial counts (positive integers), same length as observed.
        weights: Non-negative weights, same length as observed. Defaults to ones.
        verbose: If True, logs total log-likelihood.

    Returns:
        Scalar log-likelihood. Returns float("nan") if all inputs are non-finite.

    Raises:
        ValueError: If lengths of observed and estimated do not match, any weights
            are negative, all weights are zero, observed values are not integer
            counts in [0, trials], trials are not positive integers, or estimated
            probabilities are not in (0, 1).

    Examples:
        >>> calc_log_likelihood_binomial(
        ...     np.array([3, 4, 2]), np.array([0.3, 0.5, 0.25]), np.array([10, 10, 8]),
        ...     verbose=False,
        ... )
    """
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)
    trials = np.asarray(trials, dtype=float)

    if len(observed) != len(estimated):
        raise ValueError("Lengths of observed and estimated must match.")

    if weights is None:
        weights = np.ones(len(observed))
    else:
        weights = np.asarray(weights, dtype=float)

    # Remove NA quadruples
    idx = np.where(np.isfinite(observed) & np.isfinite(estimated) & np.isfinite(trials) & np.isfinite(weights))[0]
    observed = observed[idx]
    estimated = estimated[idx]
    trials = trials[idx]
    weights = weights[idx]

    if len(observed) == 0 or len(estimated) == 0 or len(weights) == 0:
        if verbose:
            logger.info("No usable data (all NA) — returning NA for log-likelihood.")
        return float("nan")

    n = len(observed)
    if len(estimated) != n or len(trials) != n or len(weights) != n:
        raise ValueError("Lengths of observed, estimated, trials, and weights must all match.")

    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero, cannot compute likelihood.")

    if np.any((observed < 0) | (observed > trials) | (observed % 1 != 0)):
        raise ValueError("observed must be integer counts between 0 and trials.")
    if np.any((trials < 1) | (trials % 1 != 0)):
        raise ValueError("trials must be positive integers.")
    if np.any((estimated <= 0) | (estimated >= 1)):
        raise ValueError("estimated probabilities must be in (0, 1).")

    ll_vec = scipy.stats.binom.logpmf(observed.astype(int), n=trials.astype(int), p=estimated)
    ll = float(np.sum(weights * ll_vec))

    if verbose:
        logger.info("Binomial log-likelihood: %.2f", ll)
    return ll


def calc_log_likelihood_gamma(
    observed: np.ndarray,
    estimated: np.ndarray,
    weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> float:
    """Calculate log-likelihood for Gamma-distributed positive continuous data.

    The shape parameter alpha is estimated via method of moments from the observed
    values (alpha = mean^2 / var). The scale parameter is per-observation:
    scale_i = estimated_i / alpha.

    Args:
        observed: Positive observed values.
        estimated: Positive expected means from the model, same length as observed.
        weights: Non-negative weights, same length as observed. Defaults to ones.
        verbose: If True, logs estimated shape and total log-likelihood.

    Returns:
        Scalar log-likelihood. Returns float("nan") if all inputs are non-finite.

    Raises:
        ValueError: If lengths of observed and estimated do not match, any weights
            are negative, all weights are zero, or any observed or estimated values
            are non-positive.

    Examples:
        >>> calc_log_likelihood_gamma(
        ...     np.array([2.5, 3.2, 1.8]), np.array([2.4, 3.0, 2.0]),
        ...     verbose=False,
        ... )
    """
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(observed) != len(estimated):
        raise ValueError("Lengths of observed and estimated must match.")

    if weights is None:
        weights = np.ones(len(observed))
    else:
        weights = np.asarray(weights, dtype=float)

    idx = np.where(np.isfinite(observed) & np.isfinite(estimated) & np.isfinite(weights))[0]
    observed = observed[idx]
    estimated = estimated[idx]
    weights = weights[idx]

    if len(observed) == 0 or len(estimated) == 0 or len(weights) == 0:
        if verbose:
            logger.info("No usable data (all NA) — returning NA for log-likelihood.")
        return float("nan")

    n = len(observed)
    if len(estimated) != n or len(weights) != n:
        raise ValueError("Lengths of observed, estimated, and weights must all match.")

    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero, cannot compute likelihood.")

    if np.any(observed <= 0):
        raise ValueError("All observed values must be strictly positive.")
    if np.any(estimated <= 0):
        raise ValueError("All estimated values must be strictly positive.")

    mu = float(np.mean(observed))
    s2 = float(np.var(observed, ddof=1))
    shape = mu**2 / s2
    scale = estimated / shape  # per-element scale vector

    if verbose:
        logger.info("Gamma shape (α) = %.2f", shape)

    ll_vec = scipy.stats.gamma.logpdf(observed, a=shape, scale=scale)
    ll = float(np.sum(weights * ll_vec))

    if verbose:
        logger.info("Gamma log-likelihood: %.2f", ll)
    return ll


def calc_log_likelihood_negbin(
    observed: np.ndarray,
    estimated: np.ndarray,
    k: Optional[float] = None,
    k_min: float = 3,
    weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> float:
    """Calculate log-likelihood for Negative Binomial-distributed count data.

    Computes the total weighted log-likelihood for count data under the Negative
    Binomial distribution. When estimated <= 0 and observed > 0, a proportional
    penalty (-observed * log(1e6)) is applied instead of -Inf.

    Args:
        observed: Non-negative integer counts (rounded internally for float safety).
        estimated: Expected means from the model, same length as observed.
        k: NB dispersion (size) parameter. If None, estimated via method of moments
            as mu^2 / (s^2 - mu); falls back to Inf (Poisson) when s^2 <= mu.
        k_min: Minimum dispersion floor applied when k is finite. Defaults to 3.
            Pass 0 to disable flooring.
        weights: Non-negative weights, same length as observed. Defaults to ones.
        verbose: If True, logs k estimation details and total log-likelihood.

    Returns:
        Scalar log-likelihood. Returns float("nan") if all inputs are non-finite.

    Raises:
        ValueError: If lengths of observed and estimated do not match, any weights
            are negative, all weights are zero, or any observed values are negative
            after rounding.

    Examples:
        >>> calc_log_likelihood_negbin(np.array([0, 5, 9]), np.array([3, 4, 5]),
        ...                            verbose=False)
        >>> calc_log_likelihood_negbin(np.array([0, 5, 9]), np.array([3, 4, 5]),
        ...                            k=1.2, verbose=False)
    """
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(observed) != len(estimated):
        raise ValueError("Lengths of observed and estimated must match.")

    if weights is None:
        weights = np.ones(len(observed))
    else:
        weights = np.asarray(weights, dtype=float)

    # Keep only finite observed/estimated/weights
    idx = np.where(np.isfinite(observed) & np.isfinite(estimated) & np.isfinite(weights))[0]
    observed = observed[idx]
    estimated = estimated[idx]
    weights = weights[idx]

    if len(observed) == 0 or len(estimated) == 0 or len(weights) == 0:
        if verbose:
            logger.info("No usable data (all NA) — returning NA for log-likelihood.")
        return float("nan")

    n = len(observed)
    if len(estimated) != n or len(weights) != n:
        raise ValueError("Lengths of observed, estimated, and weights must all match.")

    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero, cannot compute likelihood.")

    # Round to nearest integer for cross-language float safety (parquet transport
    # can produce near-integers like 2.0000000000000004)
    observed = np.round(observed)
    if np.any(observed < 0):
        raise ValueError("observed must contain non-negative integer counts.")

    # Estimate k if not supplied
    if k is None:
        mu = float(np.nanmean(observed))
        s2 = float(np.nanvar(observed, ddof=1))
        if not (np.isfinite(mu) and np.isfinite(s2) and mu > 0 and s2 > mu):
            k = np.inf
            if verbose:
                logger.info("Var = %.2f <= Mean = %.2f: using Poisson (k = Inf)", s2, mu)
        else:
            k = mu**2 / (s2 - mu)
            if verbose:
                logger.info("Estimated k = %.3f (from Var = %.3f, Mean = %.3f)", k, s2, mu)
    else:
        if verbose:
            logger.info("Using provided k = %.3f", k)

    # Apply minimum k floor when k is finite
    if np.isfinite(k) and k < k_min:
        if verbose:
            logger.info("k = %.3f < k_min = %.3f; using k_min.", k, k_min)
        k = k_min

    # Compute weighted log-likelihood with proportional penalty for zero predictions
    ll_vec = np.zeros(len(observed))

    penalty_mask = (estimated <= 0) & (observed > 0)
    normal_mask = estimated > 0

    if np.any(penalty_mask):
        ll_vec[penalty_mask] = -observed[penalty_mask] * np.log(1e6)
        if verbose:
            first_obs = int(observed[penalty_mask][0])
            logger.info(
                "NegBin: Applying proportional penalty for zero prediction (obs=%d)",
                first_obs,
            )
    # estimated <= 0 and observed == 0: ll_vec already 0 (perfect match)

    if np.any(normal_mask):
        est_safe = np.maximum(estimated[normal_mask], 1e-10)
        obs_int = observed[normal_mask].astype(int)
        if np.isinf(k):
            ll_vec[normal_mask] = scipy.stats.poisson.logpmf(obs_int, mu=est_safe)
        else:
            p_nb = k / (k + est_safe)
            ll_vec[normal_mask] = scipy.stats.nbinom.logpmf(obs_int, n=k, p=p_nb)

    ll = float(np.sum(weights * ll_vec))

    if verbose:
        msg_k = "Inf (Poisson)" if np.isinf(k) else f"{k:.3f}"
        logger.info("Negative Binomial log-likelihood (k=%s): %.2f", msg_k, ll)

    return ll


def calc_log_likelihood_normal(
    observed: np.ndarray,
    estimated: np.ndarray,
    weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> float:
    """Calculate log-likelihood for Normally-distributed continuous data.

    The residual standard deviation sigma is estimated from residuals
    (observed - estimated) using ddof=1. A Shapiro-Wilk normality test is run
    when n <= 5000, with a warning logged when p < 0.05.

    Args:
        observed: Continuous observed values.
        estimated: Model-predicted means, same length as observed.
        weights: Non-negative weights, same length as observed. Defaults to ones.
        verbose: If True, logs estimated sigma, Shapiro-Wilk p-value, and total
            log-likelihood.

    Returns:
        Scalar log-likelihood. Returns float("nan") if all inputs are non-finite.

    Raises:
        ValueError: If lengths of observed and estimated do not match, fewer than
            3 non-missing observations exist, any weights are negative, all weights
            are zero, or the residual standard deviation is non-positive.

    Examples:
        >>> ll = calc_log_likelihood_normal(
        ...     np.array([1.2, 2.8, 3.1]), np.array([1.0, 3.0, 3.2]),
        ...     verbose=False,
        ... )
    """
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(observed) != len(estimated):
        raise ValueError("Lengths of observed and estimated must match.")

    if weights is None:
        weights = np.ones(len(observed))
    else:
        weights = np.asarray(weights, dtype=float)

    # Remove NA across all three vectors
    idx = np.where(np.isfinite(observed) & np.isfinite(estimated) & np.isfinite(weights))[0]
    observed = observed[idx]
    estimated = estimated[idx]
    weights = weights[idx]

    if len(observed) == 0 or len(estimated) == 0 or len(weights) == 0:
        if verbose:
            logger.info("No usable data (all NA) — returning NA for log-likelihood.")
        return float("nan")

    n = len(observed)
    if len(estimated) != n or len(weights) != n:
        raise ValueError("Lengths of observed, estimated, and weights must all match.")

    if n < 3:
        raise ValueError("At least 3 non-missing observations are required for Normal likelihood.")

    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero, cannot compute likelihood.")

    # Estimate residual SD (ddof=1 matches R's sd())
    residuals = observed - estimated
    sigma = float(np.std(residuals, ddof=1))
    if sigma <= 0:
        raise ValueError("Standard deviation of residuals is non-positive.")

    # Shapiro-Wilk normality check (only feasible for n <= 5000)
    if n <= 5000:
        sw = scipy.stats.shapiro(residuals)
        shapiro_p = float(sw.pvalue)
        if shapiro_p < 0.05:
            logger.warning(
                "Shapiro-Wilk p = %.4f: residuals deviate from normality (p < 0.05).",
                shapiro_p,
            )
        elif verbose:
            logger.info(
                "Shapiro-Wilk p = %.4f: residuals are consistent with normality.",
                shapiro_p,
            )

    ll_vec = scipy.stats.norm.logpdf(observed, loc=estimated, scale=sigma)
    ll = float(np.sum(weights * ll_vec))

    if verbose:
        logger.info("Estimated σ = %.4f", sigma)
        logger.info("Normal log-likelihood: %.2f", ll)

    return ll


def calc_log_likelihood_poisson(
    observed: np.ndarray,
    estimated: np.ndarray,
    weights: Optional[np.ndarray] = None,
    zero_buffer: bool = True,
    verbose: bool = True,
) -> float:
    """Calculate log-likelihood for Poisson-distributed count data.

    When estimated <= 0 and observed > 0, a proportional penalty
    (-observed * log(1e6)) is applied. When zero_buffer=True, observed values
    are rounded and estimated values are floored to 1e-10.

    Args:
        observed: Non-negative integer counts.
        estimated: Expected values from the model, same length as observed.
        weights: Non-negative weights, same length as observed. Defaults to ones.
        zero_buffer: If True (default), rounds observed to integers and floors
            estimated to 1e-10. If False, enforces strict integer requirements.
        verbose: If True, logs overdispersion warnings and total log-likelihood.

    Returns:
        Scalar log-likelihood. Returns float("nan") if all inputs are non-finite.

    Raises:
        ValueError: If lengths of observed and estimated do not match, any weights
            are negative, all weights are zero, or (when zero_buffer=False) observed
            contains non-integer or negative values.

    Examples:
        >>> calc_log_likelihood_poisson(
        ...     np.array([2, 3, 4]), np.array([2.2, 2.9, 4.1]),
        ...     verbose=False,
        ... )
    """
    observed = np.asarray(observed, dtype=float)
    estimated = np.asarray(estimated, dtype=float)

    if len(observed) != len(estimated):
        raise ValueError("Lengths of observed and estimated must match.")

    if weights is None:
        weights = np.ones(len(observed))
    else:
        weights = np.asarray(weights, dtype=float)

    idx = np.where(np.isfinite(observed) & np.isfinite(estimated) & np.isfinite(weights))[0]
    observed = observed[idx]
    estimated = estimated[idx]
    weights = weights[idx]

    if len(observed) == 0 or len(estimated) == 0 or len(weights) == 0:
        if verbose:
            logger.info("No usable data (all NA) — returning NA for log-likelihood.")
        return float("nan")

    n = len(observed)
    if len(estimated) != n or len(weights) != n:
        raise ValueError("Lengths of observed, estimated, and weights must all match.")

    if np.any(weights < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero, cannot compute likelihood.")

    # Apply zero buffer if requested
    if zero_buffer:
        observed = np.round(observed)
        estimated = np.maximum(estimated, 1e-10)
    else:
        if np.any((observed < 0) | (observed % 1 != 0)):
            raise ValueError("observed must contain non-negative integer counts for Poisson.")

    if n > 1:
        mu = float(np.nanmean(observed))
        s2 = float(np.nanvar(observed, ddof=1))
        if np.isnan(mu) or mu == 0:
            logger.info("All observations are zero (or NA).")
        else:
            disp_ratio = s2 / mu
            if disp_ratio > 1.5:
                logger.warning(
                    "Var/Mean = %.2f suggests overdispersion. Consider Negative Binomial.",
                    disp_ratio,
                )

    # Compute Poisson log-likelihood with proportional penalty for zero predictions
    ll_vec = np.zeros(len(observed))

    penalty_mask = (estimated <= 0) & (observed > 0)
    normal_mask = estimated > 0

    if np.any(penalty_mask):
        ll_vec[penalty_mask] = -observed[penalty_mask] * np.log(1e6)
        if verbose:
            first_obs = int(observed[penalty_mask][0])
            logger.info(
                "Poisson: Applying proportional penalty for zero prediction (obs=%d)",
                first_obs,
            )
    # estimated <= 0 and observed == 0: ll_vec already 0 (perfect match)

    if np.any(normal_mask):
        est_safe = np.maximum(estimated[normal_mask], 1e-10)
        obs_int = observed[normal_mask].astype(int)
        ll_vec[normal_mask] = scipy.stats.poisson.logpmf(obs_int, mu=est_safe)

    ll = float(np.sum(weights * ll_vec))

    if verbose:
        logger.info("Poisson log-likelihood: %.2f", ll)
    return ll
