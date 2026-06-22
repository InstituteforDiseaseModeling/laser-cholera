# Bring Python `calc_model_likelihood` to parity with R: support per-observation confidence weights (`weights_obs_cases` / `weights_obs_deaths`)

**Repo:** InstituteforDiseaseModeling/laser-cholera
**File:** `src/laser/cholera/calc_model_likelihood.py` (function at `:511`)
**Type:** enhancement / parity fix
**Severity:** the same calibration currently scores differently on the Coiled/Dask backend vs. the local R path — a backend-dependent likelihood divergence.

## Background

MOSAIC scores candidate simulations with `calc_model_likelihood`, which exists in **two** implementations that are meant to be numerically identical:

- **R** — `MOSAIC-pkg/R/calc_model_likelihood.R` (canonical; used by the local PSOCK calibration path)
- **Python** — `laser-cholera/src/laser/cholera/calc_model_likelihood.py` (used by the Coiled/Dask worker, `MOSAIC-pkg/inst/python/mosaic_dask_worker.py:146,228`)

In **MOSAIC v0.45.3** (commit `7a265b1d`, 2026-06-19) the R function gained **per-observation confidence weights**: two optional `n_locations × n_time_steps` matrices, `weights_obs_cases` and `weights_obs_deaths`, carrying per-cell trust in `[0, 1]` (derived from surveillance source/quality — WHO / JHU / AI-observed / documented-zero / etc.). The Python version was last touched 2026-06-03 and has **no knowledge of these arguments** (`grep -rn weights_obs src/` → zero hits).

Consequently the Dask worker cannot pass the per-cell matrices and instead approximates one slice of the behavior by NaN-masking the deaths burn-in prefix (see the comment at `mosaic_dask_worker.py:183-194`: *"the engine's Python calc_model_likelihood does NOT accept the per-cell weights_obs_* matrices (MOSAIC-R-only)"*). Any run that uses non-trivial per-cell weights is therefore scored **unweighted** on Coiled and **weighted** locally — the calibrations diverge.

## Goal

Add `weights_obs_cases` / `weights_obs_deaths` to the Python `calc_model_likelihood` with **full behavioral parity** to the R implementation, so Coiled and local PSOCK produce identical log-likelihoods for identical inputs.

The R function is the **canonical specification**. Authoritative reference points (cite these when porting):

| Behavior | R reference (`R/calc_model_likelihood.R`) |
|---|---|
| New args + docstring contract | `:69-70`, `:30-40` |
| Per-location application loop | `:196-272` |
| Trivial-row detection (`.weights_obs_row_trivial`) | `:430-436` |
| Mass-preserving effective weight (`.weights_obs_effective`) | `:446-458` |
| ESS gate on effective weights | `:204-220` |
| k-coherence (k from the same weights used to score) | `:233-251` |
| Assembly (NB core only; shape terms unweighted) | `:368-382` |

## Required semantics (must match R exactly)

1. **Signature.** Add two optional args (default `None`), same grid as `obs_*`/`est_*`:
   ```python
   weights_obs_cases:  Optional[np.ndarray] = None,   # [n_locs x n_steps], values in [0,1] or NaN
   weights_obs_deaths: Optional[np.ndarray] = None,
   ```
   Validate shape == observation shape; error early on mismatch (R test #9).

2. **NB core only.** Per-cell weights affect **only** the NB cases/deaths terms. Peak-timing, peak-magnitude, cumulative, and WIS shape terms remain unweighted in v1 (R docstring `:40`). Do **not** thread per-cell weights into the shape terms.

3. **Per-location, mass-preserving effective weight.** For each location `j` and channel, replacing the current `mask_weights(weights_time, obs, est)` vector:
   ```python
   masked_wt = mask_weights(weights_time, obs, est)      # existing helper
   target_j  = masked_wt.sum()                           # mass to preserve
   wobs      = np.where(np.isfinite(wobs_row), wobs_row, 0.0)
   w_raw     = masked_wt * wobs                          # element-wise
   s         = w_raw.sum()
   w_eff     = (w_raw / s * target_j) if (s > 0 and target_j > 0) else np.zeros_like(masked_wt)
   ```
   Invariant (R test #5): `w_eff.sum() == masked_wt.sum()`. Only the *shape* of which cells are trusted changes; cross-location balance stays with `weights_location`.

4. **Trivial / byte-identity path** (R test #1, #2). If `weights_obs_* is None`, OR the row is all-`1.0` on finite-obs cells (`.weights_obs_row_trivial`), use the **existing unweighted** `mask_weights(...)` vector verbatim — do not run the renormalization. This guarantees results are byte-identical to today when no per-cell weighting is in play.

5. **ESS gate** (R test #7, #10). When per-cell weights are supplied, gate "have enough observations" on the **effective sum** over finite-obs, positive-`weights_time` cells (`sum(wobs_row[sel]) >= min_obs`, default 3) rather than the raw finite count. Trivial/None path keeps the existing raw-count gate. Cases and deaths gate **independently**.

6. **k-coherence** (R `:233-251`). Estimate the NB dispersion `k` from the **same** weight vector used for scoring: `nb_size_from_obs_weighted(obs, w_eff, k_min=...)` on the weighted path; `nb_size_from_obs_weighted(obs, weights_time, ...)` on the trivial path. (`nb_size_from_obs_weighted` already exists in the Python file.)

7. **NA handling.** `obs[j,t]` NaN → cell masked (no contribution). `wobs[j,t]` NaN with finite obs → treated as weight 0 in the product (cell effectively inherits no per-cell trust). A fully-zero weight row → that channel contributes 0 LL (R test #6).

## Insertion points in the Python file

- Signature: `calc_model_likelihood(...)` at `:511-538`.
- Per-location NB core: the cases/deaths blocks at ~`:717-738` — swap the `weights=mask_weights(...)` and the `nb_size_from_obs_weighted(..., weights_time, ...)` calls for the effective-weight / trivial-path logic above.
- `mask_weights` (`:89-110`) and `nb_size_from_obs_weighted` (`:47-86`) are reusable as-is.
- Shape-term blocks and the assembly at `:787-800` need **no** change.

## Acceptance criteria

- [ ] `calc_model_likelihood(..., weights_obs_cases=None, weights_obs_deaths=None)` is byte-identical to current output (regression).
- [ ] An all-ones weight matrix == `None` (parity with R test #1).
- [ ] A 0.5-weight cell carries exactly half a 1.0 cell's effective weight after renorm (R test #3).
- [ ] `w_eff.sum() == masked_weights_time.sum()` per location/channel (R test #5).
- [ ] Weighted NB LL equals `sum(w_eff * nbinom.logpmf(...))` by hand, with `k` from `w_eff` (R test #4).
- [ ] Shape:-mismatch matrices raise (R test #9); cases/deaths gates are independent (R test #10).
- [ ] A direct R-vs-Python cross-check on a shared fixture (same obs/est/weights_time/weights_obs/config) agrees to ~1e-8 on the total LL, weighted and unweighted.

A faithful port of the R unit tests in `MOSAIC-pkg/tests/testthat/test-calc_model_likelihood_obs_weights.R` (228 lines, 10 tests) would cover items 1–6 directly.

## Downstream (MOSAIC-pkg, NOT part of this issue)

Once the engine accepts the matrices, we will (in MOSAIC-pkg): pass the real `weights_obs_cases` / `weights_obs_deaths` from `mosaic_dask_worker.py` into `calc_model_likelihood`, and remove the deaths-prefix NaN workaround (`mosaic_dask_worker.py:183-194`) in favor of a true `weights_obs_deaths=0` prefix — matching the local PSOCK path. No engine dependency beyond the contract above.

## Note on scope

This issue intentionally ports only the **v1** R behavior (per-cell weights on the NB core; shape terms unweighted). If/when the R side extends weighting to the shape terms, a follow-up parity issue should track it.
