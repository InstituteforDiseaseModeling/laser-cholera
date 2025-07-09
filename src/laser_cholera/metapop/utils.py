from datetime import datetime
from functools import partial

import numpy as np
from laser_core.migration import distance


def get_daily_seasonality(params):
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


def get_pi_from_lat_long(params):
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


def override_helper(overrides) -> dict:
    def bool_from_string(value):
        return str(value).lower() in ("true", "1", "yes", "y", "t", "on", "enabled")

    mapping = {
        "seed": int,
        "date_start": partial(datetime.strptime, format="%Y-%m-%d"),
        "date_stop": partial(datetime.strptime, format="%Y-%m-%d"),
        "location_name": None,  # vector
        "S_j_initial": None,  # vector # TODO consider partial np.array(dtype=np.int32)
        "E_j_initial": None,  # vector
        "I_j_initial": None,  # vector
        "R_j_initial": None,  # vector
        "V1_j_initial": None,  # vector
        "V2_j_initial": None,  # vector
        "b_jt": None,  # matrix
        "d_jt": None,  # matrix
        "nu_1_jt": None,  # matrix
        "nu_2_jt": None,  # matrix
        "phi_1": float,
        "phi_2": float,
        "omega_1": float,
        "omega_2": float,
        "iota": float,
        "gamma_1": float,
        "gamma_2": float,
        "epsilon": float,
        "mu_jt": None,  # matrix
        "rho": float,
        "simga": float,
        "longitude": None,  # vector
        "latitude": None,  # vector
        "mobility_omega": float,
        "mobility_gamma": float,
        "tau_i": None,  # vector
        "beta_j0_hum": None,  # vector
        "a_1_j": None,  # vector
        "b_1_j": None,  # vector
        "a_2_j": None,  # vector
        "b_2_j": None,  # vector
        "p": int,
        "alpha_1": float,
        "alpha_2": float,
        "beta_j0_env": None,  # vector
        "theta_j": None,  # vector
        "psi_jt": None,  # matrix
        "zeta_1": float,
        "zeta_2": float,
        "kappa": float,
        "decay_days_short": float,
        "decay_days_long": float,
        "decay_shape_1": float,
        "decay_shape_2": float,
        "return": None,  # list
        "visualize": bool_from_string,
        "pdf": bool_from_string,
        "hdf5_output": bool_from_string,
        "compress": bool_from_string,
        "quiet": bool_from_string,
    }

    typed = {}
    for key, value in overrides.items():
        if key in mapping and (fn := mapping[key]) is not None:
            typed[key] = fn(value)
        else:
            typed[key] = value

    return typed
