import warnings

import numpy as np
from scipy.special import gammaln
from scipy.stats import beta
from scipy.stats import binom
from scipy.stats import gamma
from scipy.stats import norm
from scipy.stats import shapiro


def get_model_likelihood(
    obs_cases,
    sim_cases,
    obs_deaths,
    sim_deaths,
    weight_cases=None,
    weight_deaths=None,
    weights_location=None,
    weights_time=None,
    verbose=False,
):
    """
    Calculate the likelihood of the model given the observed and simulated cases and deaths.

    Parameters
    ----------
    obs_cases : np.ndarray
        Observed cases.
    sim_cases : np.ndarray
        simulated cases.
    obs_deaths : np.ndarray
        Observed deaths.
    sim_deaths : np.ndarray
        simulated deaths.
    weight_cases : np.ndarray, optional
        Weights for the cases. If None, all weights are set to 1.
    weight_deaths : np.ndarray, optional
        Weights for the deaths. If None, all weights are set to 1.
    weights_location : np.ndarray, optional
        Weights for the locations. If None, all weights are set to 1.
    weights_time : np.ndarray, optional
        Weights for the time. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The likelihood of the model given the observed and simulated cases and deaths.
    """

    # 1) Matrix dimension checks
    if (
        not isinstance(obs_cases, np.ndarray)
        or not isinstance(sim_cases, np.ndarray)
        or not isinstance(obs_deaths, np.ndarray)
        or not isinstance(sim_deaths, np.ndarray)
    ):
        raise TypeError(f"obs_* and est_* must be numpy arrays ({type(obs_cases)=}, {type(sim_cases)=}, {type(obs_deaths)=}, {type(sim_deaths)=}).")

    n_locations, n_time_steps = obs_cases.shape

    if (sim_cases.shape != obs_cases.shape) or (obs_deaths.shape != obs_cases.shape) or (sim_deaths.shape != obs_cases.shape):
        raise ValueError(
            f"obs_* and est_* must have the same dimensions (n_locations x n_time_steps). {obs_cases.shape=}, {sim_cases.shape=}, {obs_deaths.shape=}, {sim_deaths.shape=}."
        )

    # 2) Default location/time weights
    if weights_location is None:
        weights_location = np.ones(n_locations)
    if weights_time is None:
        weights_time = np.ones(n_time_steps)
    if weight_cases is None:
        weight_cases = 1
    if weight_deaths is None:
        weight_deaths = 1

    if len(weights_location) != n_locations:
        raise ValueError(f"weights_location must have length {n_locations} ({len(weights_location)=}).")
    if len(weights_time) != n_time_steps:
        raise ValueError(f"weights_time must have length {n_time_steps} ({len(weights_time)=}).")
    if np.any(weights_location < 0):
        raise ValueError(f"weights_location must be >= 0 ({weights_location.min()=}).")
    if np.any(weights_time < 0):
        raise ValueError(f"weights_time must be >= 0 ({weights_time.min()=}).")
    if np.all(weights_location == 0) or np.all(weights_time == 0):
        raise ValueError("weights_location and weights_time must not all be zero.")
    if weight_cases < 0:
        raise ValueError(f"weight_cases must be >= 0 ({weight_cases=}).")
    if weight_deaths < 0:
        raise ValueError(f"weight_deaths must be >= 0 ({weight_deaths=}).")

    # Initialize a vector to store per-location log-likelihood
    ll_locations = np.full(n_locations, np.nan)

    for j in range(n_locations):
        cases = obs_cases[j, :]
        if np.all(np.isnan(cases)):
            if verbose:
                print(f"Location {j} (observed cases): all NA — skipping.")
            continue

        mean_cases = np.nanmean(cases)
        var_cases = np.nanvar(cases, ddof=1)

        # If cases row fully NA, skip
        if np.all(np.isnan(mean_cases)) or np.all(np.isnan(var_cases)):
            if verbose:
                print(f"Location {j} (cases): all NA — skipping.")
            continue

        # Decide family for cases
        family_cases = "negbin" if (mean_cases > 0) and ((var_cases / mean_cases) >= 1.5) else "poisson"

        deaths = obs_deaths[j, :]
        if np.all(np.isnan(deaths)):
            if verbose:
                print(f"Location {j} (deaths): all NA — skipping.")
            continue

        mean_deaths = np.nanmean(deaths)
        var_deaths = np.nanvar(deaths, ddof=1)

        # If deaths row fully NA, skip
        if np.all(np.isnan(mean_deaths)) or np.all(np.isnan(var_deaths)):
            if verbose:
                print(f"Location {j} (deaths): all NA — skipping.")
            continue

        # Decide family for deaths
        family_deaths = "negbin" if (mean_deaths > 0) and ((var_deaths / mean_deaths) >= 1.5) else "poisson"

        # Calculate log-likelihood for cases
        ll_cases = calc_log_likelihood(observed=obs_cases[j,], simulated=sim_cases[j,], family=family_cases, weights=weights_time, verbose=verbose)

        ll_max_cases = calc_log_likelihood(
            observed=np.array([np.nanmax(obs_cases[j,])]),
            simulated=np.array([np.nanmax(sim_cases[j,])]),
            family="poisson",
            weights=None,
            verbose=verbose,
        )

        # Calculate log-likelihood for deaths
        ll_deaths = calc_log_likelihood(observed=obs_deaths[j,], simulated=sim_deaths[j,], family=family_deaths, weights=weights_time, verbose=False)

        ll_max_deaths = calc_log_likelihood(
            observed=np.array([np.nanmax(obs_deaths[j,])]),
            simulated=np.array([np.nanmax(sim_deaths[j,])]),
            family="poisson",
            weights=None,
            verbose=False,
        )

        # Weighted sum for location j
        ll_location_tmp = weights_location[j] * (
            weight_cases * ll_cases + weight_cases * ll_max_cases + weight_deaths * ll_deaths + weight_deaths * ll_max_deaths
        )
        ll_locations[j] = ll_location_tmp

        if verbose:
            print(f"Location {j}:")
            print(f"  Cases: var={var_cases:.2f}, mean={mean_cases:.2f} => {family_cases}, LL={ll_cases:.2f};")
            print(f"  Deaths: var={var_deaths:.2f}, mean={mean_deaths:.2f} => {family_deaths}, LL={ll_deaths:.2f};")
            print(f"  Weighted={ll_location_tmp:.2f}")

    # If everything was skipped
    if np.all(np.isnan(ll_locations)):
        if verbose:
            print("All locations skipped — returning NA.")

        return np.nan

    ll_total = np.nansum(ll_locations)

    if verbose:
        print(f"Overall total log-likelihood: {ll_total:.2f}")

    return ll_total


def calc_log_likelihood(observed, simulated, family, weights=None, **kwargs):
    """
    Calculate the log-likelihood of the observed data given the simulated data.

    Parameters
    ----------
    observed : np.ndarray
        Observed data.
    simulated : np.ndarray
        Simulated data.
    family : str
        The family of the distribution (e.g., "poisson", "negbin").
    weights : np.ndarray, optional
        Weights for the data. If None, all weights are set to 1.
    **kwargs : dict
        Additional arguments for the likelihood calculation.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """
    switch = {
        "beta": calc_log_likelihood_beta,
        "binomial": calc_log_likelihood_binomial,
        "gamma": calc_log_likelihood_gamma,
        "negbin": calc_log_likelihood_negbin,
        "normal": calc_log_likelihood_normal,
        "poisson": calc_log_likelihood_poisson,
    }

    if family in switch:
        result = switch[family](observed, simulated, weights=weights, **kwargs)
    else:
        raise ValueError(f"Unknown family: {family}")

    return result


def calc_log_likelihood_beta(observed, simulated, mean_precision=True, weights=None, verbose=True):
    """
    Calculate the log-likelihood for the Beta distribution.

    Parameters
    ----------
    observed : np.ndarray
        Observed values (must be strictly between 0 and 1).
    simulated : np.ndarray
        Simulated values (must be strictly between 0 and 1).
    mean_precision : bool, optional
        Whether to use mean-precision parameterization. Default is True.
    weights : np.ndarray, optional
        Weights for the observations. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """

    if (result := calc_log_likelihood_validation(observed, simulated, weights, verbose=verbose)) is None:
        return np.nan

    observed, simulated, weights = result

    # Beta domain checks
    if np.any((observed <= 0) | (observed >= 1)):
        raise ValueError("observed must be strictly between 0 and 1 for Beta distribution.")
    if np.any((simulated <= 0) | (simulated >= 1)):
        raise ValueError("simulated must be strictly between 0 and 1 for Beta distribution.")

    # Parameter estimation
    if mean_precision:
        residuals = observed - simulated
        sigma2 = np.var(residuals, ddof=1 if len(residuals) > 1 else 0)
        if sigma2 <= 0:
            raise ValueError("Residual variance is non-positive — cannot estimate phi.")

        mu = np.mean(observed)
        phi = (mu * (1 - mu)) / sigma2 - 1
        if phi <= 0:
            raise ValueError("simulated phi must be > 0 — data may be too dispersed or flat.")

        shape_1 = simulated * phi
        shape_2 = (1 - simulated) * phi

        if verbose:
            print(f"Mean–precision mode: simulated phi = {phi:.2f}")

    else:
        mu = np.mean(observed)
        sigma2 = np.var(observed, ddof=1 if len(observed) > 1 else 0)
        shape_1 = ((1 - mu) / sigma2 - 1 / mu) * mu**2
        shape_2 = shape_1 * (1 / mu - 1)

        if shape_1 <= 0 or shape_2 <= 0:
            raise ValueError("simulated shape parameters must be positive — check observed values.")

        if verbose:
            print(f"Standard shape mode: shape_1 = {shape_1:.2f}, shape_2 = {shape_2:.2f}")

        # Replicate for each observation
        n = len(observed)
        shape_1 = np.full(n, shape_1)
        shape_2 = np.full(n, shape_2)

    # Weighted likelihood
    ll_vec = beta.logpdf(observed, a=shape_1, b=shape_2)
    ll = np.sum(weights * ll_vec)

    if verbose:
        print(f"Beta log-likelihood: {ll:.2f}")

    return ll


def calc_log_likelihood_binomial(observed, simulated, trials, weights=None, verbose=True):
    """
    Calculate the log-likelihood for the Binomial distribution.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts (must be integers between 0 and trials).
    simulated : np.ndarray
        simulated probabilities (must be strictly between 0 and 1).
    trials : np.ndarray
        Number of trials (must be positive integers).
    weights : np.ndarray, optional
        Weights for the observations. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """

    if (result := calc_log_likelihood_validation(observed, simulated, weights, trials, verbose=verbose)) is None:
        return np.nan

    observed, simulated, trials, weights = result

    # Domain checks
    if np.any((observed < 0) | (observed > trials) | (observed % 1 != 0)):
        raise ValueError("observed must be integer counts between 0 and trials.")
    if np.any((trials < 1) | (trials % 1 != 0)):
        raise ValueError("trials must be positive integers.")
    if np.any((simulated <= 0) | (simulated >= 1)):
        raise ValueError("simulated probabilities must be in (0, 1).")

    # Weighted likelihood
    ll_vec = binom.logpmf(observed, n=trials, p=simulated)
    ll = np.sum(weights * ll_vec)

    if verbose:
        print(f"Binomial log-likelihood: {ll:.2f}")

    return ll


def calc_log_likelihood_gamma(observed, simulated, weights=None, verbose=True):
    """
    Calculate the log-likelihood for the Gamma distribution.

    Parameters
    ----------
    observed : np.ndarray
        Observed values (must be strictly positive).
    simulated : np.ndarray
        Simulated values (must be strictly positive).
    weights : np.ndarray, optional
        Weights for the observations. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """

    if (result := calc_log_likelihood_validation(observed, simulated, weights, verbose=verbose)) is None:
        return np.nan

    observed, simulated, weights = result

    # Domain checks
    if np.any(observed <= 0):
        raise ValueError("All observed values must be strictly positive.")
    if np.any(simulated <= 0):
        raise ValueError("All simulated values must be strictly positive.")

    # Parameter estimation
    mu = np.mean(observed)
    s2 = np.var(observed, ddof=1 if len(observed) > 1 else 0)
    shape = mu**2 / s2
    scale = simulated / shape

    if verbose:
        print(f"Gamma shape (⍺) = {shape:.2f}")

    # Weighted likelihood
    ll_vec = gamma.logpdf(observed, a=shape, scale=scale)
    ll = np.sum(weights * ll_vec)

    if verbose:
        print(f"Gamma log-likelihood: {ll:.2f}")

    return ll


def calc_log_likelihood_negbin(observed, simulated, k=None, weights=None, verbose=True):
    """
    Calculate the log-likelihood for the Negative Binomial distribution.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts (must be non-negative integers).
    simulated : np.ndarray
        simulated means (must be strictly positive).
    k : float, optional
        Dispersion parameter. If None, it will be simulated from the data.
    weights : np.ndarray, optional
        Weights for the observations. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """

    if (result := calc_log_likelihood_validation(observed, simulated, weights, verbose=verbose)) is None:
        return np.nan

    observed, simulated, weights = result

    # Add cushion around 0 values to avoid -Inf in log calculations
    if np.any(simulated <= 0):
        simulated = simulated.astype(np.float64)
    simulated[simulated <= 0] = np.finfo(np.float64).eps

    # Domain checks
    if np.any(observed < 0) or np.any(observed % 1 != 0):
        raise ValueError("observed must contain non-negative integer counts.")

    # Estimate k if not supplied
    if k is None:
        mu = np.mean(observed)
        s2 = np.var(observed, ddof=1 if len(observed) > 1 else 0)

        if s2 <= mu:
            if verbose:
                print(f"Var = {s2:.2f} <= Mean = {mu:.2f}: defaulting to Poisson (k = Inf)")
            k = np.inf
        else:
            k = mu**2 / (s2 - mu)
            if verbose:
                print(f"simulated k = {k:.2f} (from Var = {s2:.2f}, Mean = {mu:.2f})")
    else:
        if verbose:
            print(f"Using provided k = {k:.2f}")

    # Use Poisson if k = Inf
    if np.isinf(k):
        ll_vec = observed * np.log(simulated) - simulated - gammaln(observed + 1)
    else:
        if k < 1.5 and verbose:
            warnings.warn(f"k ({k:.2f}) < 1.5 indicates near-Poisson dispersion.")  # noqa: B028
        ll_vec = (
            gammaln(observed + k)
            - gammaln(k)
            - gammaln(observed + 1)
            + k * np.log(k / (k + simulated))
            + observed * np.log(simulated / (k + simulated))
        )

    # Weighted likelihood
    ll = np.sum(weights * ll_vec)

    if verbose:
        print(f"Negative Binomial log-likelihood: {ll:.2f}")

    return ll


def calc_log_likelihood_normal(observed, simulated, weights=None, verbose=True):
    """
    Calculate the log-likelihood for the Normal distribution.

    Parameters
    ----------
    observed : np.ndarray
        Observed values.
    simulated : np.ndarray
        Simulated values.
    weights : np.ndarray, optional
        Weights for the observations. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """

    if (result := calc_log_likelihood_validation(observed, simulated, weights, verbose=verbose)) is None:
        return np.nan

    observed, simulated, weights = result

    n = len(observed)
    if n < 3:
        raise ValueError("At least 3 non-missing observations are required for Normal likelihood.")

    # Estimate residual standard deviation
    residuals = observed - simulated
    sigma = np.std(residuals, ddof=1)  # Use ddof=1 for sample standard deviation
    if sigma <= 0:
        raise ValueError("Standard deviation of residuals is non-positive.")

    # Shapiro-Wilk normality check
    if n <= 5000:
        _shapiro_stat, shapiro_p = shapiro(residuals)
        if shapiro_p < 0.05:
            if verbose:
                print(f"Shapiro-Wilk p = {shapiro_p:.4f}: residuals deviate from normality (p < 0.05).")
        elif verbose:
            print(f"Shapiro-Wilk p = {shapiro_p:.4f}: residuals are consistent with normality.")

    # Weighted log-likelihood
    ll_vec = norm.logpdf(observed, loc=simulated, scale=sigma)
    ll = np.sum(weights * ll_vec)

    if verbose:
        print(f"simulated σ = {sigma:.4f}")
        print(f"Normal log-likelihood: {ll:.2f}")

    return ll


def calc_log_likelihood_poisson(observed, simulated, weights=None, verbose=True):
    """
    Calculate the log-likelihood for the Poisson distribution.

    Parameters
    ----------
    observed : np.ndarray
        Observed counts (must be non-negative integers).
    simulated : np.ndarray
        simulated means (must be strictly positive).
    weights : np.ndarray, optional
        Weights for the observations. If None, all weights are set to 1.
    verbose : bool, optional
        If True, print additional information.

    Returns
    -------
    float
        The log-likelihood of the observed data given the simulated data.
    """

    if (result := calc_log_likelihood_validation(observed, simulated, weights, verbose=verbose)) is None:
        return np.nan

    observed, simulated, weights = result

    # Add cushion around 0 values to avoid -Inf in log calculations
    if np.any(simulated <= 0):
        simulated = simulated.astype(np.float64)
    simulated[simulated <= 0] = np.finfo(np.float64).eps

    # Domain checks
    if np.any((observed < 0) | (observed % 1 != 0)):
        raise ValueError("observed must contain non-negative integer counts for Poisson.")

    # Check for overdispersion
    # Original code checked this, but we already return np.nan if there are no usable data
    # if len(observed) > 1:
    if (mu := np.mean(observed)) > 0:
        s2 = np.var(observed, ddof=1 if len(observed) > 1 else 0)
        if (disp_ratio := (s2 / mu)) > 1.5:
            warnings.warn(f"Var/Mean = {disp_ratio:.2f} suggests overdispersion. Consider Negative Binomial.")  # noqa: B028
    # endif

    # Weighted log-likelihood
    ll_vec = observed * np.log(simulated) - simulated - gammaln(observed + 1)
    ll = np.sum(weights * ll_vec)

    if verbose:
        print(f"Poisson log-likelihood: {ll:.2f}")

    return ll


def calc_log_likelihood_validation(observed, simulated, weights, trials=None, verbose=True):
    """
    Validate and preprocess inputs for log-likelihood calculation.
    This function performs validation and preprocessing of input arrays for
    calculating log-likelihood. It removes NaN values, checks for consistency
    in array lengths, and ensures weights are non-negative. The function
    returns the processed inputs or raises errors for invalid inputs.

    Parameters:
        observed (np.ndarray): Array of observed values.
        simulated (np.ndarray): Array of simulated values.
        weights (np.ndarray or None): Array of weights. If None, defaults to an
            array of ones with the same shape as `observed`.
        trials (np.ndarray or None, optional): Array of trial counts. If provided,
            it is also validated and returned. Defaults to None.
        verbose (bool, optional): If True, prints warnings for empty or invalid
            inputs. Defaults to True.

    Returns:
        tuple: A tuple containing the validated and preprocessed arrays:
            - (observed, simulated, weights) if `trials` is None.
            - (observed, simulated, trials, weights) if `trials` is provided.

    Raises:
        ValueError: If the lengths of `observed`, `simulated`, and `weights` do
            not match after preprocessing.
        ValueError: If any weight is negative.
        ValueError: If all weights are zero.

    Notes:
        - If all input values are NaN after preprocessing, the function returns
          None and optionally prints a warning if `verbose` is True.
    """

    # Default weights if None
    if weights is None:
        weights = np.ones_like(observed)

    # Check lengths
    n = len(observed)
    if len(simulated) != n or len(weights) != n:
        raise ValueError(f"Lengths of observed ({len(observed)}), simulated ({len(simulated)}), and weights ({len(weights)}) must all match.")

    # Remove NaNs
    if trials is None:
        mask = ~np.isnan(observed) & ~np.isnan(simulated) & ~np.isnan(weights)
    else:
        mask = ~np.isnan(observed) & ~np.isnan(simulated) & ~np.isnan(trials) & ~np.isnan(weights)
        trials = trials[mask]

    observed = observed[mask]
    simulated = simulated[mask]
    weights = weights[mask]

    # Handle empty input after NA removal
    if len(observed) == 0 or len(simulated) == 0 or len(weights) == 0:
        if verbose:
            print("No usable data (all NaN) — returning NaN for log-likelihood.")
        return None

    # Check weights
    if np.any(weights < 0):
        raise ValueError(f"All weights must be >= 0 ({weights.min()=}).")
    if np.all(weights == 0):
        raise ValueError("All weights are zero, cannot compute likelihood.")

    return (observed, simulated, weights) if trials is None else (observed, simulated, trials, weights)
