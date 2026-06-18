"""Build the Mozambique adm2 multi-admin config for docs/configurations §6.2.

Outputs:
  - docs/configurations/code/multi-admin.json.gz
  - docs/configurations/code/multi-admin.metadata.txt
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from laser.cholera.metapop.params import PseEncoder, get_parameters

REPO = Path("/Users/christopherlorton/projects/laser-fresh/laser.cholera")
INPUT_DIR = REPO / "tmp" / "laser-init" / "mozambique-adm2"
OUT_DIR = REPO / "docs" / "configurations" / "code"
OUT_JSON_GZ = OUT_DIR / "multi-admin.json.gz"
OUT_METADATA = OUT_DIR / "multi-admin.metadata.txt"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gdf = (
        gpd.read_file(INPUT_DIR / "MOZ_admin2.gpkg")
        .sort_values("nodeid")
        .reset_index(drop=True)
    )
    assert len(gdf) == 157, f"Expected 157 rows, got {len(gdf)}"
    assert gdf.crs.to_epsg() == 4326, f"Expected EPSG:4326, got {gdf.crs}"

    cxr = pd.read_csv(INPUT_DIR / "cxr.csv")

    defaults = get_parameters(None)
    moz_idx = list(defaults.location_name).index("MOZ")

    date_start = datetime(2020, 1, 1)
    date_stop = datetime(2024, 12, 31)
    nticks = (date_stop - date_start).days + 1
    assert nticks == 1827, f"Expected 1827 ticks, got {nticks}"

    npatches = 157
    location_name = gdf["adm2_name"].astype(str).tolist()

    centroids = gdf.geometry.centroid
    latitude = centroids.y.to_numpy(dtype=np.float32)
    longitude = centroids.x.to_numpy(dtype=np.float32)

    N = gdf["population"].astype(np.int64).to_numpy()
    I0 = np.zeros(npatches, dtype=np.int64)
    I0[int(np.argmax(N))] = 100
    S0 = (N - I0).astype(np.int64)
    zeros_n = np.zeros(npatches, dtype=np.int64)

    cbr_by_year = dict(zip(cxr["Time"].astype(int), cxr["CBR"].astype(float)))
    cdr_by_year = dict(zip(cxr["Time"].astype(int), cxr["CDR"].astype(float)))

    tick_years = np.array(
        [(date_start + pd.Timedelta(days=int(t))).year for t in range(nticks)],
        dtype=np.int32,
    )
    cbr_per_tick = np.array(
        [cbr_by_year[int(y)] / 1000.0 / 365.25 for y in tick_years],
        dtype=np.float32,
    )
    cdr_per_tick = np.array(
        [cdr_by_year[int(y)] / 1000.0 / 365.25 for y in tick_years],
        dtype=np.float32,
    )
    b_jt = np.broadcast_to(cbr_per_tick[:, None], (nticks, npatches)).astype(np.float32).copy()
    d_jt = np.broadcast_to(cdr_per_tick[:, None], (nticks, npatches)).astype(np.float32).copy()

    nu_1_jt = np.zeros((nticks, npatches), dtype=np.float32)
    nu_2_jt = np.zeros((nticks, npatches), dtype=np.float32)
    psi_jt = np.ones((nticks, npatches), dtype=np.float32)
    mu_jt = np.zeros((nticks, npatches), dtype=np.float32)

    def per_patch(name: str) -> np.ndarray:
        vec = np.asarray(getattr(defaults, name))
        if vec.ndim > 1:
            vec = vec.reshape(-1)
        value = vec[moz_idx]
        return np.full(npatches, value, dtype=np.float32)

    per_patch_names = [
        "beta_j0_hum",
        "a_1_j",
        "a_2_j",
        "b_1_j",
        "b_2_j",
        "theta_j",
        "beta_j0_env",
        "tau_i",
        "mu_j_baseline",
        "mu_j_slope",
        "prop_S_initial",
        "prop_E_initial",
        "prop_I_initial",
        "prop_R_initial",
        "prop_V1_initial",
        "prop_V2_initial",
    ]
    pp = {name: per_patch(name) for name in per_patch_names}

    scalar_names = [
        "phi_1",
        "phi_2",
        "omega_1",
        "omega_2",
        "iota",
        "gamma_1",
        "gamma_2",
        "epsilon",
        "rho",
        "rho_deaths",
        "sigma",
        "p",
        "alpha_1",
        "alpha_2",
        "kappa",
        "decay_days_short",
        "decay_days_long",
        "decay_shape_1",
        "decay_shape_2",
        "zeta_1",
        "zeta_2",
        "mobility_omega",
        "mobility_gamma",
        "delta_reporting_cases",
        "delta_reporting_deaths",
    ]
    scalars = {}
    for name in scalar_names:
        v = getattr(defaults, name)
        if isinstance(v, np.ndarray):
            v = v.item()
        scalars[name] = v.item() if isinstance(v, (np.integer, np.floating)) else v

    params_dict: dict = {
        "seed": 20240930,
        "date_start": date_start.strftime("%Y-%m-%d"),
        "date_stop": date_stop.strftime("%Y-%m-%d"),
        "location_name": location_name,
        "latitude": latitude,
        "longitude": longitude,
        "N_j_initial": N.astype(np.uint32),
        "S_j_initial": S0.astype(np.uint32),
        "E_j_initial": zeros_n.astype(np.uint32),
        "I_j_initial": I0.astype(np.uint32),
        "R_j_initial": zeros_n.astype(np.uint32),
        "V1_j_initial": zeros_n.astype(np.uint32),
        "V2_j_initial": zeros_n.astype(np.uint32),
        "b_jt": b_jt,
        "d_jt": d_jt,
        "nu_1_jt": nu_1_jt,
        "nu_2_jt": nu_2_jt,
        "psi_jt": psi_jt,
        "mu_jt": mu_jt,
        "epidemic_threshold": 0.0,
        "chi_endemic": 1.0,
        "chi_epidemic": 1.0,
        "mu_j_epidemic_factor": np.zeros(npatches, dtype=np.float32),
        "reported_cases": np.full((npatches, nticks), np.nan, dtype=np.float32),
        "reported_deaths": np.full((npatches, nticks), np.nan, dtype=np.float32),
        "beta_j0_tot": np.zeros(npatches, dtype=np.float32),
        "p_beta": np.zeros(npatches, dtype=np.float32),
        "psi_star_a": getattr(defaults, "psi_star_a"),
        "psi_star_b": getattr(defaults, "psi_star_b"),
        "psi_star_z": getattr(defaults, "psi_star_z"),
        "psi_star_k": getattr(defaults, "psi_star_k"),
    }
    for name, vec in pp.items():
        params_dict[name] = vec
    for name, val in scalars.items():
        params_dict[name] = val

    raw_bytes = json.dumps(params_dict, cls=PseEncoder).encode("utf-8")
    gz_bytes = gzip.compress(raw_bytes)
    OUT_JSON_GZ.write_bytes(gz_bytes)

    metadata = (
        "Source: laser-init\n"
        "Invocation: uvx --with git+https://github.com/laser-base/laser-init laser-init MOZ 2 2000 2025\n"
        "Date pulled: 2026-06-18\n"
        "Input files under tmp/laser-init/mozambique-adm2/ (NOT committed; reproduce via the invocation above):\n"
        "  - MOZ_admin2.gpkg  (157 adm2 polygons, EPSG:4326)\n"
        "  - cxr.csv  (yearly CBR/CDR 2000–2026)\n"
        "  - provenance.json  (upstream URLs + timestamps)\n"
        "  - age_dist.csv, life_exp.csv  (not used for laser-cholera)\n"
        "Date window: 2020-01-01 to 2024-12-31 (1827 ticks)\n"
        "nticks: 1827\n"
        "npatches: 157\n"
        "Notes:\n"
        "  - Per-patch values broadcast from default_parameters.json[\"MOZ\"] row across all 157 districts (not per-district calibrated).\n"
        "  - b_jt / d_jt derived from cxr.csv (country-level CBR/CDR), per-1000-per-year → per-person-per-day, broadcast.\n"
        "  - Initial seed: 100 infected in the most-populous district.\n"
        "  - Features ON: mobility, seasonality, human-to-human + environmental transmission.\n"
        "  - Features OFF: vaccination, regime switching (chi_endemic=chi_epidemic + mu_j_epidemic_factor=0), likelihood scoring.\n"
    )
    OUT_METADATA.write_text(metadata)

    params = get_parameters(str(OUT_JSON_GZ))
    print("VALIDATION_OK", params.nticks, len(params.location_name))
    print("RAW_SIZE", len(raw_bytes))
    print("GZ_SIZE", len(gz_bytes))


if __name__ == "__main__":
    main()
