# Build a multi-location country configuration

By the end of this tutorial you will have built — by hand, from a fresh `laser-init` extract — the same Mozambique adm-2 configuration (157 districts, 5 years, mobility on, seasonality on, both transmission pathways on) that ships frozen under `docs/configurations/code/multi-admin.json.gz`. You will see how scalar per-patch parameters broadcast to length-`J` vectors, how the country-level CBR / CDR series becomes a `(nticks, npatches)` rate matrix, and what happens to a single-district seed outbreak once you let the gravity-derived mobility matrix `pi_ij` start moving force-of-infection around.

## Prerequisites

You should already have walked through the [single-location tutorial](single-location.md) and be comfortable with each parameter and the canonical inert values for the features you are turning off. The numbered steps below add only one new idea at a time on top of that foundation: data, geography, mobility.

Install [`laser-init`](https://github.com/laser-base/laser-init) per its repo README and pull the Mozambique adm-2 extract:

```bash
uvx --with git+https://github.com/laser-base/laser-init laser-init MOZ 2 2000 2025
```

That invocation drops `MOZ_admin2.gpkg`, `cxr.csv`, `provenance.json`, and a couple of demographic CSVs (`age_dist.csv`, `life_exp.csv`) that this tutorial does not need. Move the extract under `tmp/laser-init/mozambique-adm2/` — the rest of this tutorial assumes that path, and `tmp/` is gitignored so the upstream data will not be committed by accident.

## Step 1 — Load and inspect the GeoPackage

`laser-init` writes admin polygons to a GeoPackage. Read it with GeoPandas, sort by `nodeid` so the row order is stable, and check the basics:

```python
from pathlib import Path
import geopandas as gpd

INPUT_DIR = Path("tmp/laser-init/mozambique-adm2")
gdf = (
    gpd.read_file(INPUT_DIR / "MOZ_admin2.gpkg")
    .sort_values("nodeid")
    .reset_index(drop=True)
)
print(len(gdf), gdf.crs)
print(gdf[["adm0_name", "adm1_name", "adm2_name", "population"]].head())
```

You should see 157 rows and `EPSG:4326` — the geometries are already in lat/lon, so the centroids you compute in the next step will land in the units that `laser.cholera` expects.

## Step 2 — Centroids, populations, names

`laser.cholera` does not consume polygons; it consumes per-patch scalars. Reduce the GeoDataFrame to four parallel length-157 sequences: latitude, longitude, initial total population, and a human-readable name per patch.

```python
latitude = gdf.geometry.centroid.y.tolist()
longitude = gdf.geometry.centroid.x.tolist()
N_j_initial = gdf["population"].astype(int).tolist()
location_name = gdf["adm2_name"].astype(str).tolist()

assert len(latitude) == len(longitude) == len(N_j_initial) == len(location_name) == 157
```

Note: these are geometric centroids. In production you may prefer population-weighted centroids so a sparsely-populated mountainous district does not anchor its representative point in an empty valley. Geometric centroids are fine for a tutorial.

## Step 3 — Seed an outbreak in the most-populous district

You want infection to start in exactly one district. Pick the most-populous one (so the seed is the place most likely to plausibly couple outwards), drop 100 infected individuals in, and balance the books for the remaining compartments:

```python
import numpy as np

idx = int(np.argmax(N_j_initial))
I_j_initial = [0] * 157
I_j_initial[idx] = 100
S_j_initial = [N - I for N, I in zip(N_j_initial, I_j_initial)]

print(f"Seed: {location_name[idx]} (population {N_j_initial[idx]:,})")
```

All other compartments (`E_j_initial`, `R_j_initial`, `V1_j_initial`, `V2_j_initial`) stay at zero in all 157 districts.

## Step 4 — Build the per-tick rate matrices from CBR/CDR

`laser-init`'s `cxr.csv` is yearly country-level crude birth and death rates per 1000 per year. The model wants per-person-per-day rates as `(nticks, npatches)` matrices. With `date_start = "2020-01-01"`, `date_stop = "2024-12-31"`, and a daily tick, `nticks = 1827`. For each tick, look up that calendar year's CBR / CDR, convert units, and broadcast to all 157 patches:

```python
import pandas as pd
from datetime import datetime

cxr = pd.read_csv(INPUT_DIR / "cxr.csv")
cbr_by_year = dict(zip(cxr["Time"].astype(int), cxr["CBR"].astype(float)))
cdr_by_year = dict(zip(cxr["Time"].astype(int), cxr["CDR"].astype(float)))

date_start = datetime(2020, 1, 1)
date_stop = datetime(2024, 12, 31)
nticks = (date_stop - date_start).days + 1  # 1827

tick_years = np.array(
    [(date_start + pd.Timedelta(days=int(t))).year for t in range(nticks)]
)
cbr_per_tick = np.array([cbr_by_year[int(y)] / 1000.0 / 365.25 for y in tick_years], dtype=np.float32)
cdr_per_tick = np.array([cdr_by_year[int(y)] / 1000.0 / 365.25 for y in tick_years], dtype=np.float32)

b_jt = np.broadcast_to(cbr_per_tick[:, None], (nticks, 157)).astype(np.float32).copy()
d_jt = np.broadcast_to(cdr_per_tick[:, None], (nticks, 157)).astype(np.float32).copy()
print(b_jt.shape, b_jt.dtype)  # (1827, 157) float32
```

Because `cxr.csv` is country-level, every district shares the same rate at each tick — the columns of `b_jt` and `d_jt` are identical. That is a deliberate simplification of this configuration, not a limitation of the model: nothing stops you from supplying district-specific rates if you have them.

## Step 5 — Broadcast per-patch scalars from the MOZ defaults row

The bundled `default_parameters.json` carries one value per country for each per-patch quantity. Pull the Mozambique row and broadcast each per-patch parameter 157 times. Scalars (period, recovery rates, mobility exponents, etc.) copy verbatim:

```python
from laser.cholera.metapop.params import get_parameters

defaults = get_parameters(None)
moz_idx = list(defaults.location_name).index("MOZ")

per_patch_names = [
    "beta_j0_hum", "a_1_j", "a_2_j", "b_1_j", "b_2_j",
    "theta_j", "beta_j0_env", "tau_i",
    "mu_j_baseline", "mu_j_slope",
    "prop_S_initial", "prop_E_initial", "prop_I_initial",
    "prop_R_initial", "prop_V1_initial", "prop_V2_initial",
]
def per_patch(name):
    vec = np.asarray(getattr(defaults, name)).reshape(-1)
    return np.full(157, vec[moz_idx], dtype=np.float32)

pp = {name: per_patch(name) for name in per_patch_names}
```

This is **not** a calibrated Mozambique fit — every district sees the same `beta_j0_hum`, the same WASH coverage, the same seasonal amplitudes. It is a working country-shaped configuration that you can iterate from.

The scalar parameters — `phi_1`, `phi_2`, `omega_1`, `omega_2`, `iota`, `gamma_1`, `gamma_2`, `epsilon`, `sigma`, `p`, `alpha_1`, `alpha_2`, `kappa`, `decay_days_short`, `decay_days_long`, `decay_shape_1`, `decay_shape_2`, `zeta_1`, `zeta_2`, `mobility_omega`, `mobility_gamma`, `delta_reporting_cases`, `delta_reporting_deaths` — are copied through unchanged from `defaults`.

## Step 6 — Turn vaccination and regime switching off

This is the same pattern you used in the single-location tutorial, scaled to 157 patches. Set both partners of every "off" pair so each disabled feature is fully inert:

```python
nu_1_jt = np.zeros((nticks, 157), dtype=np.float32)
nu_2_jt = np.zeros((nticks, 157), dtype=np.float32)
V1_j_initial = np.zeros(157, dtype=np.uint32)
V2_j_initial = np.zeros(157, dtype=np.uint32)

# Regime switching: both chi == 1.0 AND mu_j_epidemic_factor == 0.
chi_endemic = 1.0
chi_epidemic = 1.0
mu_j_epidemic_factor = np.zeros(157, dtype=np.float32)

# Likelihood scoring: leave reported_* as NaN so no score is computed.
reported_cases = np.full((157, nticks), np.nan, dtype=np.float32)
reported_deaths = np.full((157, nticks), np.nan, dtype=np.float32)
```

The environmental suitability driver `psi_jt` is left at a `(1827, 157)` ones matrix — the canonical inert value documented in the single-location tutorial — even though `beta_j0_env` is non-zero here, because the model's env-to-human term initialises off `(psi - psi_bar) / psi_bar` and needs a strictly positive constant baseline to be well-defined.

## Step 7 — Validate, build, run

Assemble the whole parameter dictionary, hand it to `get_parameters` for validation, then construct a `Model` with the canonical 13-component pipeline (the same one `tests/test_docs_configurations.py` uses). To keep the tutorial fast, override `nticks` to ~60 days before calling `run()`:

```python
from datetime import timedelta
from laser.cholera.metapop import (
    Analyzer, Census, DerivedValues, Environmental, EnvToHuman, Exposed,
    HumanToHuman, Infectious, Parameters, Recorder, Recovered,
    Susceptible, Vaccinated,
)
from laser.cholera.metapop.model import Model

params_dict = {
    "seed": 20240930,
    "date_start": "2020-01-01",
    "date_stop": "2024-12-31",
    "location_name": location_name,
    "latitude": latitude,
    "longitude": longitude,
    "N_j_initial": N_j_initial,
    "S_j_initial": S_j_initial,
    "E_j_initial": [0] * 157,
    "I_j_initial": I_j_initial,
    "R_j_initial": [0] * 157,
    "V1_j_initial": V1_j_initial,
    "V2_j_initial": V2_j_initial,
    "b_jt": b_jt, "d_jt": d_jt,
    "nu_1_jt": nu_1_jt, "nu_2_jt": nu_2_jt,
    "psi_jt": np.ones((nticks, 157), dtype=np.float32),
    "mu_jt": np.zeros((nticks, 157), dtype=np.float32),
    "epidemic_threshold": 0.0,
    "chi_endemic": chi_endemic, "chi_epidemic": chi_epidemic,
    "mu_j_epidemic_factor": mu_j_epidemic_factor,
    "reported_cases": reported_cases,
    "reported_deaths": reported_deaths,
    **pp,
    # ... plus the scalar names listed in Step 5, copied from `defaults`.
}

params = get_parameters(params_dict)

# Keep the tutorial run cheap: shrink to 60 ticks before running.
params.nticks = 60
params.date_stop = params.date_start + timedelta(days=59)
for name in ("b_jt", "d_jt", "nu_1_jt", "nu_2_jt", "psi_jt"):
    setattr(params, name, getattr(params, name)[:60, :])

model = Model(params, name="tutorial-mozambique-adm2")
model.components = [
    Susceptible, Exposed, Recovered, Infectious, Vaccinated, Census,
    HumanToHuman, EnvToHuman, Environmental, DerivedValues,
    Analyzer, Recorder, Parameters,
]
model.run()
```

If validation fails, `get_parameters` raises with a message naming the offending field — read it carefully; nine times out of ten it is a length mismatch (a per-patch vector with 156 or 158 entries instead of 157) or a matrix transposed the wrong way.

## Step 8 — See what mobility did

The whole point of this configuration is that mobility couples the 157 districts. With `tau_i > 0` and a non-trivial `pi_ij`, force-of-infection in the seed district leaks outwards along the gravity-derived coupling matrix. First, the country-wide infected trajectory:

```python
import matplotlib.pyplot as plt

total_I = model.people.I_sym.sum(axis=1) + model.people.I_asym.sum(axis=1)
plt.plot(total_I)
plt.xlabel("tick (day)")
plt.ylabel("total infected, summed across 157 districts")
plt.show()
```

Then the spatial picture at the end of the run. Attach end-of-run infected counts back to the GeoDataFrame and plot a choropleth:

```python
gdf["I_end"] = (model.people.I_sym[-1, :] + model.people.I_asym[-1, :])
gdf.plot(column="I_end", legend=True)
plt.title("Infected by district, last tick")
plt.show()
```

You should see infection radiating out from the seed district into its neighbours. That spread is what `pi_ij` does: laser-cholera derives it internally from `latitude`, `longitude`, `N_j_initial`, `mobility_omega`, and `mobility_gamma`, so by providing those inputs in Steps 2 and 5 you already wired in a working gravity model. ([`laser.core`](https://github.com/InstituteforDiseaseModeling/laser/tree/main/src/laser_core) ships standalone migration / gravity models that document the math; you do **not** need to call them directly here, because `laser.cholera` computes `pi_ij` for you from the same inputs.)

## Where to next

- [Configurations › multi-admin](../configurations/multi-admin.md) — the same configuration you just built, shipped frozen with full provenance.
- [Reference › Geography and mobility](../reference/parameters/geography-and-mobility.md) — `tau_i`, `mobility_omega`, `mobility_gamma`, and the `latitude` / `longitude` columns that drive `pi_ij`.
- [Explanation › Mobility](../explanation/mobility.md) (wave 5 stub) — how `pi_ij` is computed from the gravity inputs.
- [How-to: configure mobility](../how-to/configure-mobility.md) (wave 4 stub) — recipe-style coverage of `tau_i`, `mobility_omega`, `mobility_gamma`.
