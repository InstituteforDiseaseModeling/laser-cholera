"""Spring likelihood functions for scoring cholera model fits against observed data.

Translated from calc_model_likelihood.R. Scores model fits using Negative Binomial
(NB) time-series log-likelihood per location and outcome (cases, deaths) with a
weighted method-of-moments dispersion estimate and a k_min floor.

Optional shape terms are enabled by setting their weight > 0: peak timing (Normal),
peak magnitude (log-Normal with adaptive sigma), cumulative progression (NB at
cumulative fractions), and Weighted Interval Score (WIS). All weights default to 0.

Shape terms are internally T-normalized so that weight parameters share a common
scale: weight=0.25 means the term contributes roughly 25% as much as the NB core.

The `epidemic_peaks` dataset required by peak shape terms and legacy helpers must be
supplied as a pandas DataFrame with columns ``iso_code`` and ``peak_date``, either
via ``config["epidemic_peaks"]`` or as an explicit function argument.

Translation complete. Here's a summary of the key design decisions:

**Indexing**: R uses 1-based indices; Python uses 0-based. All `peak_indices` stored and passed as 0-based. Window slices use `[w_start:w_end]` with `w_end = peak_idx + 15` (exclusive) to match R's `(peak_idx-14):(peak_idx+14)` inclusive range.

**NB distribution**: R's `dnbinom(x, size=k, mu=mu)` → `scipy.stats.nbinom.logpmf(x, n=k, p=k/(k+mu))`. The `p` conversion is applied wherever NB distributions are evaluated.

**`MOSAIC::calc_log_likelihood`**: Implemented locally as `_calc_log_likelihood_nb` since it's not in the provided R source.

**`MOSAIC::epidemic_peaks`**: Replaced with a `epidemic_peaks` parameter (pandas DataFrame) passed either via `config["epidemic_peaks"]` in the main function or as an explicit argument to the legacy helpers.

**`verbose`**: R's `message()` calls translated to `logger.info()` — the `verbose` flag is respected for the summary messages; internal loop logs are always emitted at INFO level per project convention.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats

logger = logging.getLogger(__name__)


def nb_size_from_obs_weighted(
    x: np.ndarray,
    w: np.ndarray,
    k_min: float = 3,
    k_max: float = 1e5,
) -> float:
    """Estimate NB dispersion (size) from observed data via weighted method-of-moments.

    Uses Bessel-corrected weighted variance (V1² / (V1² − V2) normalisation, where
    V1 = Σw and V2 = Σw²) to avoid underestimating variance with small or
    unequal-weight samples.

    Args:
        x: Observed count time series (1-D array).
        w: Non-negative weights, same length as x.
        k_min: Minimum dispersion floor. Defaults to 3.
        k_max: Maximum dispersion cap. Defaults to 1e5.

    Returns:
        Estimated NB size parameter k, clipped to [k_min, k_max]. Returns np.inf if
        fewer than two finite, positive-weight observations exist or if the variance
        does not exceed the mean (Poisson / sub-Poisson regime).
    """
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(ok):
        return np.inf
    x_ok = x[ok]
    w_ok = w[ok]
    sw = float(np.sum(w_ok))
    sw2 = float(np.sum(w_ok**2))
    m = float(np.sum(w_ok * x_ok) / sw)
    denom = sw - sw2 / sw
    if denom > 0:
        v = float(np.sum(w_ok * (x_ok - m) ** 2) / denom)
    else:
        v = float(np.sum(w_ok * (x_ok - m) ** 2) / sw)
    if not (np.isfinite(m) and np.isfinite(v) and m > 0 and v > m):
        return np.inf
    k = (m * m) / (v - m)
    return float(np.clip(k, k_min, k_max))


def mask_weights(
    w: np.ndarray,
    obs_vec: np.ndarray,
    est_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Zero out weights where observations or estimates are non-finite.

    Args:
        w: Weight vector, same length as obs_vec.
        obs_vec: Observed values.
        est_vec: Optional estimated values; non-finite entries also zero out weights.

    Returns:
        Copy of w with weights zeroed where obs_vec (or est_vec) is non-finite.
    """
    w2 = w.copy()
    bad = ~np.isfinite(obs_vec)
    if est_vec is not None:
        bad = bad | ~np.isfinite(est_vec)
    if np.any(bad):
        w2[bad] = 0.0
    return w2


def _calc_log_likelihood_nb(
    observed: np.ndarray,
    estimated: np.ndarray,
    weights: np.ndarray,
    k: float,
    k_min: float = 3,
) -> float:
    """Compute weighted Negative Binomial log-likelihood for a time series.

    When k is infinite the distribution collapses to Poisson. The k_min floor is
    applied before evaluation to prevent near-Poisson collapse on low-variance series.

    Args:
        observed: Observed counts (rounded to integers internally).
        estimated: Estimated means (must be non-negative).
        weights: Per-timestep weights; entries that are non-positive are excluded.
        k: NB dispersion (size). Use np.inf for Poisson.
        k_min: Minimum dispersion floor applied before evaluation. Defaults to 3.

    Returns:
        Weighted sum of NB (or Poisson) log-PMF values. Returns 0.0 if no valid
        observations exist.
    """
    mask = np.isfinite(observed) & np.isfinite(estimated) & (weights > 0)
    if not np.any(mask):
        return 0.0

    obs_m = np.round(observed[mask]).astype(int)
    est_m = np.maximum(estimated[mask], 1e-10)
    w_m = weights[mask]

    k_eff = max(k_min, k) if np.isfinite(k) else np.inf

    if np.isinf(k_eff):
        ll_vals = scipy.stats.poisson.logpmf(obs_m, mu=est_m)
    else:
        p_nb = k_eff / (k_eff + est_m)
        ll_vals = scipy.stats.nbinom.logpmf(obs_m, n=k_eff, p=p_nb)

    return float(np.sum(w_m * ll_vals))


def _calc_peak_timing_from_indices(
    est_vec: np.ndarray,
    peak_indices: list,
    sigma_peak_time: float = 1,
    timestep_to_weeks: float = 7,
) -> float:
    """Score estimated peak timing against known peak indices using a Normal prior.

    For each known peak index, locates the model peak within a ±14-step window and
    scores the timing offset (in weeks) with a Normal(0, sigma_peak_time) log-PDF.

    Args:
        est_vec: Estimated time series for one location (1-D array).
        peak_indices: 0-based indices of known epidemic peaks.
        sigma_peak_time: SD in weeks for the Normal timing prior. Defaults to 1.
        timestep_to_weeks: Conversion factor from timesteps to weeks (7 for daily
            data, 1 for weekly data). Defaults to 7.

    Returns:
        Sum of Normal log-PDFs for timing offsets across all peaks. Returns 0.0 if
        no peaks are provided or no window is large enough to evaluate.
    """
    ll_total = 0.0
    n_ts = len(est_vec)
    for peak_idx in peak_indices:
        w_start = max(0, peak_idx - 14)
        w_end = min(n_ts, peak_idx + 15)
        if (w_end - w_start) > 2:
            est_peak_idx = w_start + int(np.argmax(est_vec[w_start:w_end]))
            time_diff = (est_peak_idx - peak_idx) / timestep_to_weeks
            ll_total += float(scipy.stats.norm.logpdf(time_diff, loc=0, scale=sigma_peak_time))
    return ll_total


def _calc_peak_magnitude_from_indices(
    obs_vec: np.ndarray,
    est_vec: np.ndarray,
    peak_indices: list,
    sigma_peak_log: float = 0.5,
) -> float:
    """Score estimated peak magnitude against known peak indices using an adaptive log-Normal prior.

    For each known peak index, scores the log-ratio of estimated to observed peak
    magnitude within a ±14-step window. The sigma shrinks as sqrt(100 / obs_peak) for
    large peaks, imposing tighter constraints where the data is more reliable.

    Args:
        obs_vec: Observed time series for one location (1-D array).
        est_vec: Estimated time series for one location (1-D array).
        peak_indices: 0-based indices of known epidemic peaks.
        sigma_peak_log: Base SD on the log scale. Defaults to 0.5.

    Returns:
        Sum of Normal log-PDFs for log-ratio peak magnitudes. Returns 0.0 if no
        peaks qualify or observed/estimated peaks are non-positive.
    """
    ll_total = 0.0
    n_ts = len(obs_vec)
    for peak_idx in peak_indices:
        w_start = max(0, peak_idx - 14)
        w_end = min(n_ts, peak_idx + 15)
        if (w_end - w_start) > 2:
            obs_peak_val = float(np.nanmax(obs_vec[w_start:w_end]))
            est_peak_val = float(np.nanmax(est_vec[w_start:w_end]))
            if np.isfinite(obs_peak_val) and np.isfinite(est_peak_val) and obs_peak_val > 0 and est_peak_val > 0:
                adaptive_sigma = sigma_peak_log * np.sqrt(100.0 / max(obs_peak_val, 100.0))
                ll_total += float(
                    scipy.stats.norm.logpdf(
                        np.log(est_peak_val) - np.log(obs_peak_val),
                        loc=0,
                        scale=adaptive_sigma,
                    )
                )
    return ll_total


def calc_multi_peak_timing_ll(
    obs_vec: np.ndarray,
    est_vec: np.ndarray,
    iso_code: Optional[str] = None,
    date_start=None,
    date_stop=None,
    sigma_peak_time: float = 1,
    epidemic_peaks=None,
) -> float:
    """Compute peak timing log-likelihood using epidemic peaks data (legacy interface).

    Matches epidemic peak dates to the time series via a date sequence, then scores
    estimated peak timing within ±14-step windows using a Normal prior.

    Args:
        obs_vec: Observed time series for one location (1-D array).
        est_vec: Estimated time series for one location (1-D array).
        iso_code: ISO country code used to look up peaks in epidemic_peaks.
        date_start: Start date of the time series (string or date-like).
        date_stop: End date of the time series (string or date-like).
        sigma_peak_time: SD in weeks for the Normal timing prior. Defaults to 1.
        epidemic_peaks: pandas DataFrame with columns ``iso_code`` and ``peak_date``.
            Returns 0.0 if None.

    Returns:
        Sum of Normal log-PDFs for timing offsets. Returns 0.0 if required inputs
        are missing, no peaks are found, or the date sequence cannot be built.
    """
    if epidemic_peaks is None or iso_code is None or date_start is None or date_stop is None:
        return 0.0

    loc_peaks = epidemic_peaks[epidemic_peaks["iso_code"] == iso_code]
    if len(loc_peaks) == 0:
        return 0.0

    date_seq = pd.date_range(start=date_start, end=date_stop, freq="D")
    timestep_to_weeks = 7
    if len(date_seq) != len(obs_vec):
        date_seq = pd.date_range(start=date_start, end=date_stop, freq="W")
        if len(date_seq) != len(obs_vec):
            return 0.0
        timestep_to_weeks = 1

    peak_indices = []
    for peak_date in loc_peaks["peak_date"]:
        idx = int(np.argmin(np.abs(date_seq - pd.Timestamp(peak_date))))
        if 0 <= idx < len(obs_vec):
            peak_indices.append(idx)

    if not peak_indices:
        return 0.0

    return _calc_peak_timing_from_indices(est_vec, peak_indices, sigma_peak_time, timestep_to_weeks)


def calc_multi_peak_magnitude_ll(
    obs_vec: np.ndarray,
    est_vec: np.ndarray,
    iso_code: Optional[str] = None,
    date_start=None,
    date_stop=None,
    sigma_peak_log: float = 0.5,
    epidemic_peaks=None,
) -> float:
    """Compute peak magnitude log-likelihood using epidemic peaks data (legacy interface).

    Matches epidemic peak dates to the time series via a date sequence, then scores
    estimated peak magnitudes within ±14-step windows using an adaptive log-Normal prior.

    Args:
        obs_vec: Observed time series for one location (1-D array).
        est_vec: Estimated time series for one location (1-D array).
        iso_code: ISO country code used to look up peaks in epidemic_peaks.
        date_start: Start date of the time series (string or date-like).
        date_stop: End date of the time series (string or date-like).
        sigma_peak_log: Base SD on the log scale. Defaults to 0.5.
        epidemic_peaks: pandas DataFrame with columns ``iso_code`` and ``peak_date``.
            Returns 0.0 if None.

    Returns:
        Sum of Normal log-PDFs for log-ratio peak magnitudes. Returns 0.0 if required
        inputs are missing, no peaks are found, or the date sequence cannot be built.
    """
    if epidemic_peaks is None or iso_code is None or date_start is None or date_stop is None:
        return 0.0

    loc_peaks = epidemic_peaks[epidemic_peaks["iso_code"] == iso_code]
    if len(loc_peaks) == 0:
        return 0.0

    date_seq = pd.date_range(start=date_start, end=date_stop, freq="D")
    if len(date_seq) != len(obs_vec):
        date_seq = pd.date_range(start=date_start, end=date_stop, freq="W")
        if len(date_seq) != len(obs_vec):
            return 0.0

    peak_indices = []
    for peak_date in loc_peaks["peak_date"]:
        idx = int(np.argmin(np.abs(date_seq - pd.Timestamp(peak_date))))
        if 0 <= idx < len(obs_vec):
            peak_indices.append(idx)

    if not peak_indices:
        return 0.0

    return _calc_peak_magnitude_from_indices(obs_vec, est_vec, peak_indices, sigma_peak_log)


def ll_cumulative_progressive_nb(
    obs_vec: np.ndarray,
    est_vec: np.ndarray,
    timepoints: np.ndarray = np.array([0.25, 0.5, 0.75, 1.0]),  # noqa: B008
    k_data: Optional[float] = None,
    weights_time: Optional[np.ndarray] = None,
    k_fallback: float = 10.0,
) -> float:
    """Compute cumulative-progression NB log-likelihood at fractional timepoints.

    Evaluates the NB log-PMF at cumulative sums of obs/est at each fractional
    timepoint. The NB size is scaled proportionally to the number of summed
    timesteps (k * end_idx), reflecting variance scaling of summed NB variables.
    Each timepoint contribution is normalized by end_idx to yield a per-observation
    LL, making it scale-compatible with other shape components at assembly.

    Args:
        obs_vec: Observed count time series (1-D array).
        est_vec: Estimated count time series (1-D array).
        timepoints: Fractional timepoints at which cumulative sums are evaluated.
            Defaults to [0.25, 0.5, 0.75, 1.0].
        k_data: Data-driven NB dispersion from nb_size_from_obs_weighted. If None
            or non-finite, falls back to k_fallback.
        weights_time: Retained for API compatibility with the R version; not used
            in the cumulative sum computation.
        k_fallback: Fallback k when k_data is unavailable. Defaults to 10.0.

    Returns:
        Mean per-observation LL across valid timepoints. Returns 0.0 if no valid
        timepoints exist.
    """
    n = len(obs_vec)

    vals = []

    for tp in timepoints:
        end_idx = int(np.clip(round(n * tp), 1, n))

        cum_k = k_data * end_idx if (k_data is not None and np.isfinite(k_data)) else k_fallback

        o_cum = float(np.nansum(obs_vec[:end_idx]))
        e_cum = float(np.nansum(est_vec[:end_idx]))

        if not (np.isfinite(o_cum) and np.isfinite(e_cum)):
            continue

        if e_cum <= 0 and o_cum > 0:
            vals.append((-round(o_cum) * np.log(1e6)) / end_idx)
            continue

        e_cum = max(e_cum, 1e-10)
        p_nb = cum_k / (cum_k + e_cum)
        ll_tp = float(scipy.stats.nbinom.logpmf(round(o_cum), n=cum_k, p=p_nb))

        if not np.isfinite(ll_tp):
            vals.append((-round(o_cum) * np.log(1e6)) / end_idx)
            continue

        vals.append(ll_tp / end_idx)

    if not vals:
        return 0.0
    return float(np.mean(vals))


def compute_wis_parametric_row(
    y: np.ndarray,
    est: np.ndarray,
    w_time: np.ndarray,
    probs: np.ndarray,
    k_use: float,
) -> float:
    """Compute Weighted Interval Score (WIS) for a single time-series row.

    Uses NB (or Poisson when k_use is infinite) quantile functions evaluated at each
    time step to score the observed series against the estimated series. The final
    score is the weighted average over time of interval scores across all symmetric
    quantile pairs, plus a median absolute error term.

    Args:
        y: Observed time series (1-D array).
        est: Estimated means (1-D array, same length as y).
        w_time: Per-timestep weights (non-negative).
        probs: Quantile levels. Symmetric pairs (p, 1-p) are matched for interval
            scoring; the 0.5 quantile is used for the median AE term.
        k_use: NB dispersion. Use np.inf to fall back to Poisson.

    Returns:
        WIS score (lower is better). Returns np.nan if all observations are
        non-finite or total weight is zero.
    """
    if not (np.any(np.isfinite(y)) and np.any(np.isfinite(est))):
        return np.nan

    w_use = w_time.copy().astype(float)
    bad = ~np.isfinite(y) | ~np.isfinite(est)
    if np.any(bad):
        w_use[bad] = 0.0
    if np.sum(w_use) == 0:
        return np.nan

    est_eval = np.maximum(est, 1e-12)

    def qfun(p_val: float) -> np.ndarray:
        if np.isinf(k_use):
            return scipy.stats.poisson.ppf(p_val, mu=est_eval)
        p_nb = k_use / (k_use + est_eval)
        return scipy.stats.nbinom.ppf(p_val, n=k_use, p=p_nb)

    probs_sorted = np.sort(np.unique(probs))
    has_med = np.any(np.abs(probs_sorted - 0.5) < 1e-8)
    mae_term = 0.0
    if has_med:
        q_med = qfun(0.5)
        mae_term = 0.5 * float(np.sum(np.abs(y - q_med) * w_use) / np.sum(w_use))

    lowers = probs_sorted[probs_sorted < 0.5]
    uppers = probs_sorted[probs_sorted > 0.5]

    pairs = []
    for p in lowers:
        complement = 1.0 - p
        match_idx = np.where(np.abs(uppers - complement) < 1e-8)[0]
        if len(match_idx) > 0:
            pairs.append((p, uppers[match_idx[0]]))
        else:
            pairs.append((p, uppers[int(np.argmin(np.abs(uppers - complement)))]))

    k_pairs = len(pairs)
    sum_is = 0.0
    for p_l, p_u in pairs:
        q_l = qfun(p_l)
        q_u = qfun(p_u)
        alpha = 1.0 - (p_u - p_l)
        width = q_u - q_l
        under = np.maximum(0.0, q_l - y) * (2.0 / alpha)
        over = np.maximum(0.0, y - q_u) * (2.0 / alpha)
        contrib = float(np.sum((width + under + over) * w_use) / np.sum(w_use))
        sum_is += (alpha / 2.0) * contrib

    return float((mae_term + sum_is) / (k_pairs + 0.5))


def calc_model_likelihood(
    obs_cases: np.ndarray,
    est_cases: np.ndarray,
    obs_deaths: np.ndarray,
    est_deaths: np.ndarray,
    weight_cases: float = 1.0,
    weight_deaths: float = 1.0,
    weights_location: Optional[np.ndarray] = None,
    weights_time: Optional[np.ndarray] = None,
    config: Optional[dict] = None,
    nb_k_min_cases: float = 3,
    nb_k_min_deaths: float = 3,
    verbose: bool = False,
    weight_peak_timing: float = 0,
    weight_peak_magnitude: float = 0,
    weight_cumulative_total: float = 0,
    weight_wis: float = 0,
    sigma_peak_time: float = 1,
    sigma_peak_log: float = 0.5,
    wis_quantiles: np.ndarray = np.array([0.025, 0.25, 0.5, 0.75, 0.975]),  # noqa: B008
    cumulative_timepoints: np.ndarray = np.array([0.25, 0.5, 0.75, 1.0]),  # noqa: B008
) -> float:
    """Compute total model log-likelihood against observed cases and deaths.

    Scores model fits using a weighted Negative Binomial time-series log-likelihood
    per location and outcome. The NB dispersion k is estimated from observed data via
    weighted method-of-moments with a k_min floor, making it a property of the
    observation process rather than the model fit.

    Optional shape terms (all off by default) are T-normalized so that a weight of
    0.25 contributes roughly 25% as much as the NB core:

    - **Peak timing**: Normal(0, sigma_peak_time) on the timing offset in weeks.
    - **Peak magnitude**: log-Normal with adaptive sigma on the observed/estimated
      peak ratio.
    - **Cumulative progression**: NB on cumulative sums at fractional timepoints.
    - **WIS**: Negated Weighted Interval Score using NB quantile functions.

    The ``epidemic_peaks`` DataFrame (required for peak terms) must be provided via
    ``config["epidemic_peaks"]`` with columns ``iso_code`` and ``peak_date``.

    Assembly formula per location j::

        ll_loc = wc * NB_cases + wd * NB_deaths
               + (N_obs/N_peaks)    * w_pt  * (wc*pt_c  + wd*pt_d)
               + (N_obs/N_peaks)    * w_pm  * (wc*pm_c  + wd*pm_d)
               + (N_obs/N_eval_pts) * w_cum * (wc*cum_c + wd*cum_d)
               + (N_obs/N_quant)    * w_wis * (wc*wis_c + wd*wis_d)

    Args:
        obs_cases: Observed case counts, shape (n_locations, n_time_steps).
        est_cases: Estimated case counts, shape (n_locations, n_time_steps).
        obs_deaths: Observed death counts, shape (n_locations, n_time_steps).
        est_deaths: Estimated death counts, shape (n_locations, n_time_steps).
        weight_cases: Scalar weight multiplier for all case components. Defaults to 1.
        weight_deaths: Scalar weight multiplier for all death components. Defaults to 1.
        weights_location: Non-negative location weights, length n_locations. Defaults
            to ones.
        weights_time: Non-negative time weights, length n_time_steps. Defaults to ones.
        config: Optional dict with keys ``location_name`` (list of ISO codes),
            ``date_start``, ``date_stop``, and optionally ``epidemic_peaks`` (DataFrame).
        nb_k_min_cases: Minimum NB dispersion floor for cases. Defaults to 3.
        nb_k_min_deaths: Minimum NB dispersion floor for deaths. Defaults to 3.
        verbose: If True, logs per-location component summaries at INFO level.
        weight_peak_timing: Weight for peak timing term (T-normalized). Defaults to 0.
        weight_peak_magnitude: Weight for peak magnitude term (T-normalized). Defaults to 0.
        weight_cumulative_total: Weight for cumulative progression term. Defaults to 0.
        weight_wis: Weight for WIS term (T-normalized). Defaults to 0.
        sigma_peak_time: SD in weeks for the peak timing Normal prior. Defaults to 1.
        sigma_peak_log: Base SD on log-scale for peak magnitude prior. Defaults to 0.5.
        wis_quantiles: Quantile levels for WIS scoring. Defaults to
            [0.025, 0.25, 0.5, 0.75, 0.975].
        cumulative_timepoints: Fractional timepoints for cumulative progression.
            Defaults to [0.25, 0.5, 0.75, 1.0].

    Returns:
        Scalar total log-likelihood. Returns -np.inf if the total is non-finite and
        np.nan if all locations contribute NA (e.g., all have too few observations).

    Raises:
        ValueError: If any input is not a 2-D array, dimensions are inconsistent,
            estimated values are negative, weights are negative, or weight vectors
            sum to zero.
    """
    if (
        not (isinstance(obs_cases, np.ndarray) and obs_cases.ndim == 2)
        or not (isinstance(est_cases, np.ndarray) and est_cases.ndim == 2)
        or not (isinstance(obs_deaths, np.ndarray) and obs_deaths.ndim == 2)
        or not (isinstance(est_deaths, np.ndarray) and est_deaths.ndim == 2)
    ):
        raise ValueError("All inputs must be 2-D arrays (n_locations x n_time_steps).")

    if np.any(est_cases < 0) or np.any(est_deaths < 0):
        raise ValueError("Estimated values must be non-negative.")

    n_locations, n_time_steps = obs_cases.shape

    if (
        est_cases.shape != (n_locations, n_time_steps)
        or obs_deaths.shape != (n_locations, n_time_steps)
        or est_deaths.shape != (n_locations, n_time_steps)
    ):
        raise ValueError("All arrays must have the same shape (n_locations x n_time_steps).")

    if weights_location is None:
        weights_location = np.ones(n_locations)
    else:
        weights_location = np.asarray(weights_location, dtype=float)
    if weights_time is None:
        weights_time = np.ones(n_time_steps)
    else:
        weights_time = np.asarray(weights_time, dtype=float)

    if len(weights_location) != n_locations:
        raise ValueError("weights_location must match n_locations.")
    if len(weights_time) != n_time_steps:
        raise ValueError("weights_time must match n_time_steps.")
    if np.any(weights_location < 0) or np.any(weights_time < 0):
        raise ValueError("All weights must be >= 0.")
    if np.sum(weights_location) == 0 or np.sum(weights_time) == 0:
        raise ValueError("weights_location and weights_time must not all be zero.")

    # --- precompute peak indices per location (once, not per call) ---
    peak_indices_by_loc = None
    timestep_to_weeks = 7
    if (weight_peak_timing > 0 or weight_peak_magnitude > 0) and config is not None:
        location_names = config.get("location_name")
        date_start_cfg = config.get("date_start")
        date_stop_cfg = config.get("date_stop")
        epidemic_peaks = config.get("epidemic_peaks")

        if location_names is not None and date_start_cfg is not None and date_stop_cfg is not None and epidemic_peaks is not None:
            date_seq = pd.date_range(start=date_start_cfg, end=date_stop_cfg, freq="D")
            if len(date_seq) != n_time_steps:
                date_seq = pd.date_range(start=date_start_cfg, end=date_stop_cfg, freq="W")
                if len(date_seq) != n_time_steps:
                    date_seq = None
                else:
                    timestep_to_weeks = 1

            if date_seq is not None:
                logger.info("Precomputing peak indices for %d locations.", n_locations)
                peak_indices_by_loc = [[] for _ in range(n_locations)]
                for j_pk in range(n_locations):
                    iso_code = location_names[j_pk] if j_pk < len(location_names) else None
                    if iso_code is None:
                        continue
                    loc_peaks = epidemic_peaks[epidemic_peaks["iso_code"] == iso_code]
                    if len(loc_peaks) == 0:
                        continue
                    for peak_date in loc_peaks["peak_date"]:
                        idx = int(np.argmin(np.abs(date_seq - pd.Timestamp(peak_date))))
                        if 0 <= idx < n_time_steps:
                            peak_indices_by_loc[j_pk].append(idx)

    # --- main loop ---
    ll_locations = np.full(n_locations, np.nan)

    min_obs_for_likelihood = 3

    for j in range(n_locations):
        obs_c = obs_cases[j, :]
        est_c = est_cases[j, :]
        obs_d = obs_deaths[j, :]
        est_d = est_deaths[j, :]

        have_cases = int(np.sum(np.isfinite(obs_c))) >= min_obs_for_likelihood
        have_deaths = int(np.sum(np.isfinite(obs_d))) >= min_obs_for_likelihood

        # k is estimated from observed data (property of observation noise, not fit quality)
        k_c = nb_size_from_obs_weighted(obs_c, weights_time, k_min=nb_k_min_cases) if have_cases else np.inf
        k_d = nb_size_from_obs_weighted(obs_d, weights_time, k_min=nb_k_min_deaths) if have_deaths else np.inf

        ll_cases = (
            _calc_log_likelihood_nb(
                observed=obs_c,
                estimated=est_c,
                weights=mask_weights(weights_time, obs_c, est_c),
                k=k_c,
                k_min=nb_k_min_cases,
            )
            if have_cases
            else 0.0
        )
        ll_deaths = (
            _calc_log_likelihood_nb(
                observed=obs_d,
                estimated=est_d,
                weights=mask_weights(weights_time, obs_d, est_d),
                k=k_d,
                k_min=nb_k_min_deaths,
            )
            if have_deaths
            else 0.0
        )

        ll_peak_time_c = ll_peak_time_d = 0.0
        ll_peak_mag_c = ll_peak_mag_d = 0.0

        if (weight_peak_timing > 0 or weight_peak_magnitude > 0) and peak_indices_by_loc is not None:
            loc_peak_idx = peak_indices_by_loc[j]
            if loc_peak_idx:
                if weight_peak_timing > 0:
                    if have_cases:
                        ll_peak_time_c = _calc_peak_timing_from_indices(est_c, loc_peak_idx, sigma_peak_time, timestep_to_weeks)
                    if have_deaths:
                        ll_peak_time_d = _calc_peak_timing_from_indices(est_d, loc_peak_idx, sigma_peak_time, timestep_to_weeks)
                if weight_peak_magnitude > 0:
                    if have_cases:
                        ll_peak_mag_c = _calc_peak_magnitude_from_indices(obs_c, est_c, loc_peak_idx, sigma_peak_log)
                    if have_deaths:
                        ll_peak_mag_d = _calc_peak_magnitude_from_indices(obs_d, est_d, loc_peak_idx, sigma_peak_log)

        ll_cum_tot_c = ll_cum_tot_d = 0.0
        if weight_cumulative_total > 0:
            if have_cases:
                ll_cum_tot_c = ll_cumulative_progressive_nb(obs_c, est_c, cumulative_timepoints, k_c, weights_time)
            if have_deaths:
                ll_cum_tot_d = ll_cumulative_progressive_nb(obs_d, est_d, cumulative_timepoints, k_d, weights_time)

        ll_wis_cases = ll_wis_deaths = 0.0
        if weight_wis > 0:
            if have_cases:
                wis_c = compute_wis_parametric_row(obs_c, est_c, weights_time, wis_quantiles, k_use=k_c)
                if np.isfinite(wis_c):
                    ll_wis_cases = -wis_c
            if have_deaths:
                wis_d = compute_wis_parametric_row(obs_d, est_d, weights_time, wis_quantiles, k_use=k_d)
                if np.isfinite(wis_d):
                    ll_wis_deaths = -wis_d

        # N_obs: timesteps with at least one finite observation
        n_obs = int(np.sum(np.isfinite(obs_c) | np.isfinite(obs_d)))

        n_peaks_j = len(peak_indices_by_loc[j]) if peak_indices_by_loc is not None else 0
        n_wis_quant = len(wis_quantiles)
        n_cum_points = len(cumulative_timepoints)

        # Scale factors: N_obs / N_component_obs (T-normalization)
        peak_scale = (n_obs / n_peaks_j) if n_peaks_j > 0 else 0.0
        wis_scale = (n_obs / n_wis_quant) if n_wis_quant > 0 else 0.0
        cum_scale = (n_obs / n_cum_points) if n_cum_points > 0 else 0.0

        ll_loc_core = weight_cases * ll_cases + weight_deaths * ll_deaths
        ll_loc_peaks = peak_scale * weight_peak_timing * (
            weight_cases * ll_peak_time_c + weight_deaths * ll_peak_time_d
        ) + peak_scale * weight_peak_magnitude * (weight_cases * ll_peak_mag_c + weight_deaths * ll_peak_mag_d)
        ll_loc_cum = cum_scale * weight_cumulative_total * (weight_cases * ll_cum_tot_c + weight_deaths * ll_cum_tot_d)
        ll_loc_wis = wis_scale * weight_wis * (weight_cases * ll_wis_cases + weight_deaths * ll_wis_deaths)

        ll_loc_total = ll_loc_core + ll_loc_peaks + ll_loc_cum + ll_loc_wis

        if not np.isfinite(ll_loc_total):
            ll_locations[j] = -np.inf
            continue

        ll_locations[j] = weights_location[j] * ll_loc_total
        logger.info(
            "Location %d: core=%.2f | peaks=%.2f | cum=%.2f | wis=%.2f -> weighted=%.2f",
            j + 1,
            ll_loc_core,
            ll_loc_peaks,
            ll_loc_cum,
            ll_loc_wis,
            float(ll_locations[j]),
        )

    if np.all(np.isnan(ll_locations)):
        if verbose:
            logger.info("All locations contributed NA — returning nan.")
        return float("nan")

    ll_total = float(np.nansum(ll_locations))
    if not np.isfinite(ll_total):
        ll_total = -np.inf
    logger.info("Overall total log-likelihood: %.2f", ll_total)
    return ll_total
