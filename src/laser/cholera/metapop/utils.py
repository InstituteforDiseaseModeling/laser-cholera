"""Shared metapop helpers: seasonality, gravity-model mobility, and CLI-override coercion.

Three responsibilities live here:

- [`get_daily_seasonality`][laser.cholera.metapop.utils.get_daily_seasonality]
  — bakes the per-tick / per-patch human-to-human transmission
  seasonality envelope from Fourier coefficients (`a_1_j`, `b_1_j`,
  `a_2_j`, `b_2_j`, period `p`).
- [`get_pi_from_lat_long`][laser.cholera.metapop.utils.get_pi_from_lat_long]
  — builds the row-stochastic spatial-connectivity matrix `pi_ij`
  using a gravity model over patch lat/long, attenuated by
  `mobility_omega` / `mobility_gamma`.
- [`override_helper`][laser.cholera.metapop.utils.override_helper] and
  the [`UnknownOverrideKey`][laser.cholera.metapop.utils.UnknownOverrideKey]
  exception, used by `cli_run` to type-coerce and strictly validate
  `--over key:value` overrides against the parameter schema.
"""

import difflib
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from laser.core.migration import distance

if TYPE_CHECKING:
    from laser.cholera.metapop.params import PropertySetEx


class UnknownOverrideKey(ValueError):
    """Raised when `override_helper` receives a key that is not in its mapping.

    Subclass of `ValueError` so callers that catch `ValueError` continue
    to work. CLI front-ends (`cli_run`) catch this specific subclass and
    re-raise as `click.UsageError` for a clean CLI presentation; other
    `ValueError`s (notably the `_cli_unsupported` rejections raised for
    valid-but-non-scalar parameter keys) propagate as-is.
    """


def get_daily_seasonality(params: "PropertySetEx") -> np.ndarray:
    """Build the per-tick, per-patch human-to-human transmission seasonality envelope.

    Computes `beta_j0_hum * (1 + a_1_j cos(2pi t/p) + b_1_j sin(2pi t/p)
    + a_2_j cos(4pi t/p) + b_2_j sin(4pi t/p))`, returning a
    `(nticks, npatches)` `float32` array. `t` is 1-indexed to match the
    R reference implementation.

    Args:
        params: A `PropertySetEx` with `beta_j0_hum`, `a_1_j`, `b_1_j`,
            `a_2_j`, `b_2_j`, `p` (period in ticks), and `nticks`.

    Returns:
        A `(nticks, npatches)` `float32` array of the multiplicative
        seasonality envelope. Consumed once by `HumanToHuman.__init__`
        and stored on `patches.beta_jt_human`.
    """
    beta_j0_hum = params.beta_j0_hum
    a1 = params.a_1_j
    b1 = params.b_1_j
    a2 = params.a_2_j
    b2 = params.b_2_j
    p = params.p
    t = np.arange(0, params.nticks) + 1  # R is 1-indexed, so we start at 1

    seasonality = (
        beta_j0_hum
        * (
            1.0
            + a1[None, :] * np.cos(2 * np.pi * t / p)[:, None]
            + b1[None, :] * np.sin(2 * np.pi * t / p)[:, None]
            + a2[None, :] * np.cos(4 * np.pi * t / p)[:, None]
            + b2[None, :] * np.sin(4 * np.pi * t / p)[:, None]
        )
    ).astype(np.float32)

    return seasonality


def get_pi_from_lat_long(params: "PropertySetEx") -> np.ndarray:
    """Build the row-stochastic spatial-connectivity matrix `pi_ij` from a gravity model.

    Computes `x_ij = N_j^omega * d_ij^(-gamma)` for each origin/dest
    pair (excluding the self pair `j == i`), then row-normalizes:
    `pi_ij = x_ij / sum_j(x_ij)`. The diagonal is zero (no
    self-mobility). The migrating fraction `tau_i` is *not* applied
    here; it's factored in at runtime inside
    [`HumanToHuman.__call__`][laser.cholera.metapop.humantohuman.HumanToHuman]
    so the migrating fraction can vary per patch without rebuilding
    `pi_ij`.

    Special-cases the single-location configuration: `distance()`
    returns a 0-dim scalar that's promoted to a `(1, 1)` array so
    downstream array operations keep working; the result is then
    trivially `[[0.0]]`.

    Args:
        params: A `PropertySetEx` with `latitude` (npatches,),
            `longitude` (npatches,), `mobility_omega`, `mobility_gamma`,
            and the seed initial-population vectors
            (`S_j_initial`, `E_j_initial`, `I_j_initial`, `R_j_initial`,
            `V1_j_initial`, `V2_j_initial`) used to compute `N_j`.

    Returns:
        An `(npatches, npatches)` `float32` row-stochastic matrix.
    """
    # x <- D; x[,] <- NA
    # for (i in 1:length(N_orig)) {
    #   for (j in 1:length(N_dest)) {

    #     x[i,j] <- (N_dest[j]^params[k,'omega']) * (D[i,j]+0.001)^(-params[k,'gamma'])

    #   }
    # }

    # for (i in 1:length(N_orig)) {
    #   for (j in 1:length(N_dest)) {

    #     # M_hat[i,j] <- params[k,'theta'] * N_orig[i] * (x[i,j]/sum(x[i,]))
    #     M_hat[i,j] <- x[i,j]/sum(x[i,])

    #   }
    # }

    d = distance(params.latitude, params.longitude, params.latitude, params.longitude)
    # Handle single location case in which return from distance() is a scalar
    if not d.shape:
        # Convert to (1, 1) array.
        d = np.array([[d]], dtype=d.dtype)
    x = np.zeros_like(d, dtype=np.float32)
    omega = params.mobility_omega
    gamma = params.mobility_gamma
    N = params.S_j_initial + params.E_j_initial + params.I_j_initial + params.R_j_initial + params.V1_j_initial + params.V2_j_initial
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if j == i:
                continue
            # gravity model uses origin and destination populations
            # we'll incorporate the destination population now
            # and the effective origin population, including tau (migrating fraction) at runtime
            x[i, j] = np.power(N[j], omega) * np.power(d[i, j], -gamma)

    m_hat = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        row_sum = np.sum(x[i, :])
        for j in range(x.shape[1]):
            if j == i:
                continue
            m_hat[i, j] = x[i, j] / row_sum

    return m_hat


def _cli_unsupported(key):
    """Build a coercer that rejects CLI overrides for non-scalar parameters.

    The returned function raises a `ValueError` describing why the key
    cannot be set from `--over` and pointing at the escape hatches
    (`--params` JSON file, or `get_parameters(mods=...)` from Python).
    Used in `override_helper`'s mapping for every known parameter whose
    value is a vector, matrix, DataFrame, or other non-trivially-string-
    coercible payload.

    Args:
        key: Parameter name; embedded into the error message so the user
            sees which override they need to drop.

    Returns:
        A unary callable that always raises `ValueError`.
    """

    def reject(_value):
        raise ValueError(
            f"Parameter '{key}' cannot be set via --over (requires a non-scalar "
            f"value). Pass a parameters JSON file via --params, or call "
            f"get_parameters(mods={{'{key}': ...}}) from Python."
        )

    return reject


def override_helper(overrides: dict) -> dict:
    """Coerce stringly-typed `--over` parameter overrides to their runtime types.

    Called from `cli_run` against the parsed `--over key:value` payload
    only (CLI flags like `--seed` / `--hdf5-output` are typed by click
    itself and never reach this function). Each entry in the mapping
    table is a unary callable invoked on the raw string value:

    - ``int``-mapped keys (e.g. ``seed``, ``p``, ``delta_reporting_cases``)
      → `int(value)`.
    - ``float``-mapped keys (e.g. ``phi_1``, ``sigma``, ``rho``,
      ``chi_endemic``) → `float(value)`.
    - ``date_start`` / ``date_stop`` → `datetime` parsed from `"%Y-%m-%d"`.
    - "CLI-unsupported" keys (vectors, matrices, DataFrames such as
      ``S_j_initial``, ``b_jt``, ``epidemic_peaks``, ``return``) — the
      coercer immediately raises `ValueError`. These are valid model
      parameters but cannot meaningfully be passed as a CLI string;
      use `--params` or `get_parameters(mods=...)` instead.
    - Unknown keys raise `UnknownOverrideKey` (a `ValueError` subclass)
      with a `difflib`-derived "did you mean" suggestion when a close
      match exists.

    The mapping is the source of truth for which parameter names exist;
    every key in `default_parameters.json` is represented, plus the
    `cli_run`-only sentinels (none, after the `--hdf5-output` / `--compress`
    promotions in the same change).

    Args:
        overrides: Mapping of override key → raw string value, as parsed
            from `--over` tokens by `cli_run`.

    Returns:
        A new dict with the same keys as the input, values coerced per
        the table.

    Raises:
        UnknownOverrideKey: When an override key is not in the mapping.
        ValueError: When an override key is in the mapping but flagged
            as CLI-unsupported (vector / matrix / DataFrame).

    Example:
        >>> from laser.cholera.metapop.utils import override_helper
        >>> typed = override_helper({"phi_1": "0.65", "seed": "42"})
        >>> typed["phi_1"] == 0.65 and typed["seed"] == 42
        True
        >>> override_helper({"b_jt": "anything"})
        Traceback (most recent call last):
            ...
        ValueError: Parameter 'b_jt' cannot be set via --over...
        >>> override_helper({"date_strat": "2024-01-01"})
        Traceback (most recent call last):
            ...
        laser.cholera.metapop.utils.UnknownOverrideKey: Unknown override key 'date_strat'. Did you mean 'date_start'?
    """

    # `datetime.strptime` is a C function that rejects keyword arguments, so
    # `functools.partial` against it would TypeError; wrap in a lambda instead.
    def _parse_date(value):
        return datetime.strptime(value, "%Y-%m-%d")  # noqa: DTZ007

    _unsupported_keys = (
        # Vectors (length-npatches or length-ncompartments)
        "location_name",
        "N_j_initial",
        "S_j_initial",
        "E_j_initial",
        "I_j_initial",
        "R_j_initial",
        "V1_j_initial",
        "V2_j_initial",
        "prop_S_initial",
        "prop_E_initial",
        "prop_I_initial",
        "prop_R_initial",
        "prop_V1_initial",
        "prop_V2_initial",
        "longitude",
        "latitude",
        "tau_i",
        "beta_j0_hum",
        "beta_j0_env",
        "beta_j0_tot",
        "p_beta",
        "a_1_j",
        "a_2_j",
        "b_1_j",
        "b_2_j",
        "theta_j",
        "psi_star_a",
        "psi_star_b",
        "psi_star_z",
        "psi_star_k",
        "epidemic_threshold",
        "mu_j_baseline",
        "mu_j_slope",
        "mu_j_epidemic_factor",
        "nu_jt_sources",
        # Matrices (npatches × nticks)
        "b_jt",
        "d_jt",
        "nu_1_jt",
        "nu_2_jt",
        "psi_jt",
        "mu_jt",
        "reported_cases",
        "reported_deaths",
        # Structured
        "epidemic_peaks",
        "return",
    )

    mapping = {
        # --- scalars: CLI-coercible ---
        "seed": int,
        "date_start": _parse_date,
        "date_stop": _parse_date,
        "phi_1": float,
        "phi_2": float,
        "omega_1": float,
        "omega_2": float,
        "iota": float,
        "gamma_1": float,
        "gamma_2": float,
        "epsilon": float,
        "rho": float,
        "rho_deaths": float,
        "sigma": float,
        "chi_endemic": float,
        "chi_epidemic": float,
        "mobility_omega": float,
        "mobility_gamma": float,
        "p": int,
        "alpha_1": float,
        "alpha_2": float,
        "zeta_1": float,
        "zeta_2": float,
        "zeta_ratio": float,
        "kappa": float,
        "decay_days_short": float,
        "decay_days_long": float,
        "decay_days_spread": int,
        "decay_shape_1": float,
        "decay_shape_2": float,
        "delta_reporting_cases": int,
        "delta_reporting_deaths": int,
        # --- known but CLI-unsupported (vectors / matrices / DataFrames) ---
        **{k: _cli_unsupported(k) for k in _unsupported_keys},
    }

    typed = {}
    for key, value in overrides.items():
        if key not in mapping:
            hint = difflib.get_close_matches(key, mapping, n=1, cutoff=0.6)
            suggestion = f" Did you mean '{hint[0]}'?" if hint else ""
            raise UnknownOverrideKey(f"Unknown override key '{key}'.{suggestion}")
        typed[key] = mapping[key](value)

    return typed
