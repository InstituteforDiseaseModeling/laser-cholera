
# Changelog

## Unreleased

- 📝🧪 Wave-1 Author phase for the standalone documentation rewrite. Fill in all ten parameter group pages under `docs/reference/parameters/` — `run-identity.md`, `initial-populations.md`, `vital-dynamics.md`, `vaccination.md`, `disease-progression.md`, `reporting.md`, `regime-switching.md`, `geography-and-mobility.md`, `human-transmission.md`, `environmental-transmission.md` — each with a brief intro, a quick-reference table, and a per-parameter entry that gives the definition, shape, dtype, valid range, consumer code, related parameters, and the off-value that disables the feature the parameter gates. Rewrite `docs/reference/parameters/index.md` as the master index: a grouped table per group (linking into the relevant page anchors) and an alphabetical table over all 76 top-level fields of `default_parameters.json`. Carry forward three documented decisions surfaced during the Verify phase: (1) collapsing the endemic-vs-epidemic regime requires `chi_endemic == chi_epidemic` AND `mu_j_epidemic_factor == 0` (the plan's "set `epidemic_threshold = 0` or large" half-measure is insufficient on its own); (2) the seasonal-harmonic off-value for `b_1_j` / `b_2_j` is zero (vector of zeros), not "any non-zero" — they are sine amplitudes, never denominators, so zeroing them alongside `a_1_j` / `a_2_j` collapses the harmonic bracket to 1 cleanly. Add `tests/test_docs_param_coverage.py` (5 tests, given-when-then style with docstrings, uses `logger` from `laser.cholera`): walks the group pages and asserts every `default_parameters.json` key is documented in exactly one page, that no documented parameter is missing from the JSON (typo / stale-entry catcher), that every name in `params.py`'s `scalars` / `arrays` lists is documented, and that all ten group pages exist as files. The test is the source of truth for parameter-reference completeness — failures should be fixed in the docs.
- 📝🔧 Wave-1 scaffold for the standalone documentation rewrite. Add an explicit Diátaxis-shaped `nav:` block to `mkdocs.yml` (Tutorials / How-to / Reference / Explanation / Configurations as top-level entries) and retire the `literate-nav` plugin in favour of hand-listed reference entries; `docs/_gen_reference.py` no longer emits a `SUMMARY.md`. Create the new docs subtrees `docs/tutorials/`, `docs/how-to/`, `docs/reference/parameters/`, `docs/explanation/`, `docs/configurations/` (plus `docs/configurations/code/`) and stub every new page with a one-line H1 and a "coming in wave N" admonition so `mkdocs build --strict` stays green.
- 🔧 Align the canonical default seed to `20240930`: bump the `metapop --seed` default in `src/laser/cholera/metapop/model.py` from `20241107` to `20240930`, and update the `docs/usage.md` CLI and Python examples (previously `20240101`). No tests pin `20241107` or `20240101`, so test changes are unnecessary; `misc/perf_baseline.py`'s perf seed is intentionally left alone.

## 0.12.5 (unreleased)

- Add src/laser/cholera/calc_log_likelihood_distributions.py: Python translation of calc_log_likelihood_distributions.R (Beta, Binomial, Gamma, NegBin, Normal, Poisson)
- Rename src/laser/cholera/spring_likelihood.py → calc_model_likelihood.py
- Add tests/test_calc_log_likelihood_negbin.py: Python translation of test_calc_log_likelihood_negbin.R; update import to calc_log_likelihood_distributions
- Add tests/test_calc_model_likelihood.py: Python translation of test_calc_model_likelihood.R
- Add tests/test_calc_model_likelihood_extreme.py: Python translation of test_calc_model_likelihood_extreme.R
- Add tests/test_calc_model_likelihood_reference.py: Python translation of test_calc_model_likelihood_reference.R
- Add tests/test_compute_wis_parametric_row.py: Python translation of test_compute_wis_parametric_row.R
- Add tests/test_ll_cumulative_progressive_nb.py: Python translation of test_ll_cumulative_progressive_nb.R
- Add tests/test_nb_size_from_obs_weighted.py: Python translation of test_nb_size_from_obs_weighted.R
- Update src/laser/cholera/calc_model_likelihood.py: replace the `config` dict argument on `calc_model_likelihood` with explicit `epidemic_peaks` (DataFrame with `iso_code`, `peak_date`, `loc_idx` columns), `date_start`, and `date_stop` kwargs.
- Update src/laser/cholera/metapop/params.py: ingestion of `epidemic_peaks` now asserts each `iso_code` is in `location_name` and appends a `loc_idx` column mapping each row to its simulation location index.
- Remove HDF5 config-parameter loading from src/laser/cholera/metapop/params.py (`load_hdf5_parameters`, `load_compressed_hdf5_parameters`, `load_hdf5` and the `.h5`/`.hdf`/`.hdf5` entries in `get_parameters` dispatch). HDF5 *output* via `recorder.py` is unaffected, as is the `hdf5_output` flag in `utils.py`. Add a parametrized regression test in tests/test_params.py confirming HDF5 suffixes are now rejected by `get_parameters`.
- Add tests/test_recorder.py: 20-test coverage of `Recorder` — init/model identity, `check()` warn-on-missing for `people`/`patches`, per-tick gating in `__call__` (final-tick timing, `hdf5_output` + `return` quadrants), `outdir` routing, `compress` → `.h5.gz`, low-level `save_hdf5_parameters` / `save_compressed_hdf5_parameters` (groups, datasets, underscore + method filtering, `AttributeError` on missing frames), and an end-to-end happy-path round-trip.
- Docs: clean up the `calc_model_likelihood` module docstring (drop "Spring" prefix and translation-progress narrative), reorder its `Args:` block to match the call signature, fix grammar ("a"→"an"), document the all-zero `weights_location` / `weights_time` `ValueError`, add a runnable `Example` doctest for the core-NB call, and clarify in `calc_multi_peak_timing_ll` / `calc_multi_peak_magnitude_ll` that `loc_idx` is not required (with `Raises: KeyError` from pandas indexing). Replace the `docs/usage.rst` placeholder with a likelihood-focused usage guide (doctest-verified) covering the minimal call, the shape-term weights, and the analyzer-integration recipe.
- Port the R upstream commit's in-window peak filter to `calc_model_likelihood`, `calc_multi_peak_timing_ll`, and `calc_multi_peak_magnitude_ll`: peak rows whose `peak_date` falls outside `[date_start, date_stop]` are now dropped before index assignment instead of being clamped by `np.argmin` to t=0 or t=n-1. Add `test_out_of_window_peaks_are_filtered` and `test_mixed_window_peaks_only_in_window_counted` in tests/test_calc_model_likelihood.py to pin the new behavior.
- Update tests/test_params.py: add tests covering `epidemic_peaks` ingestion (list-of-dicts and dict-of-lists → DataFrame, optional/absent case, `loc_idx` mapping correctness, unknown-ISO `AssertionError`) and `validate_parameters` enforcement of `iso_code` and `peak_date` columns.
- 🦺 Add the symmetric `check_key(mapping, key, message)` helper to `src/laser/cholera/metapop/utils.py` and use it to collapse the 56 `if key not in mapping: raise ValueError(...)` call sites across the 10 compartment modules. Same shape as `check_attr` but for mapping-membership: validates that required `params` entries (or any container supporting `in`) are present, raises a precise `ValueError` if not. Tests / lint / docs all green; total assertion-style validation in the metapop pipeline is now one-line per check.
- 🦺 Introduce `check_attr(obj, attr, message)` in `src/laser/cholera/metapop/utils.py` and use it to collapse the wordy `if not hasattr(obj, attr): raise AttributeError(...)` pattern at all 49 call sites across the 10 compartment modules into one-liners. Each modified site has the same semantics as the prior expanded form — the helper just hoists the `if`/`raise` into a function so the caller reads as plain validation: `check_attr(model, "people", "Susceptible: model needs to have an 'people' attribute.")`. The helper's docstring carries a runnable doctest showing both the no-op and the raise. Imports in the 10 component files were updated to include the helper (auto-sorted by ruff). 271 tests + 2 doctests pass; lint clean. The corresponding `if "X" not in mapping: raise ValueError(...)` pattern is still wordy and could be collapsed the same way with a `check_key` helper — pending a separate ask.
- 🦺📝 Complete the H4 sweep across the compartment modules. The first pass converted `params.py`'s 80 input-validation asserts to `if/raise ValueError(...)`; this pass extends the same treatment to every component file under `src/laser/cholera/metapop/`. 105 validation asserts converted: 49 `assert hasattr(X, "Y"), msg` → `if not hasattr(X, "Y"): raise AttributeError(msg)`, 47 single-line `assert "X" in Y, msg` → `if "X" not in Y: raise ValueError(msg)`, 9 multi-line / line-wrapped `assert "X" in Y, (\n ... )` variants of the same. The 16 asserts that remain are legitimate invariants kept exactly as the original audit recommended: 13 post-binomial-draw `assert np.all(X >= 0)` non-negativity checks across the 7 compartments that draw stochastically, plus 3 internal shape-equality checks (2 in `EnvToHuman.__init__` post-allocation consistency, 1 in `calculate_coupling`). Every modified `Raises:` docstring block was rewritten to enumerate `AttributeError` / `ValueError` instead of the generic `AssertionError`. Running with `python -O` no longer silently disables the parameter / model-attribute checks in any component. Closes the H4 residual flagged in `assessment.md`'s Status update.
- 🔧 Remove the Read the Docs scaffolding now that docs ship from GitHub Pages. Deleted `.readthedocs.yml`. `docs/index.md` had a paragraph saying the site was mid-migration and that prose still rendered on RTD — rewritten to describe the actual MkDocs / Material / mkdocstrings / GitHub Pages pipeline. `mkdocs.yml` had a stale comment about prose pages migrating from `.rst` to `.md`; trimmed. `doc-conversion.md §8`'s `[ ] Decide whether to keep the RTD project` checkbox marked done with the decision recorded. References to RTD in `CHANGELOG.md` / `assessment.md` / `doc-conversion.md` left intact as historical / planning record. External `laser-cholera.readthedocs.io` links can be updated as encountered; nothing in-repo points at them now.
- 🔧 Consolidate the docs dependency list onto the `[docs]` extra in `pyproject.toml` and delete `docs/requirements.txt` (the two lists were already byte-equivalent; this just removes the duplicate). `tox.ini`'s `[testenv:docs]` switched from `deps = -r{toxinidir}/docs/requirements.txt` to the canonical `extras = docs` field, and `.readthedocs.yml`'s install block switched from `requirements: docs/requirements.txt` to `pip` install with `extra_requirements: [docs]`. The GitHub Actions docs workflow was already using `uv run --extra docs …` so no change there. Note: `.readthedocs.yml` still references the now-deleted `docs/conf.py` from the pre-MkDocs era — flagged in `assessment.md` for a separate decision about RTD's fate. Closes L7.
- 📝 Resolve assessment item L6 (RST `.. math::` blocks in docstrings). All 8 occurrences converted to `$$…$$` arithmatex/MathJax syntax: 1 in `environmental.py:map_suitability_to_decay`, 1 in `humantohuman.py:HumanToHuman.__call__`, 5 in `derivedvalues.py:DerivedValues.__call__`, and 1 each in `derivedvalues.py:calculate_spatial_hazard` and `calculate_coupling`. The originally-cited file list missed three of these — picked them up during a `grep -rn '\.\. math::' src/` sweep. Affected docstrings switched to `r"""…"""` so the LaTeX source reads with single backslashes in the file rather than escaped doubles. While editing the `map_suitability_to_decay` docstring (the only numpydoc-style one in the package), converted its `Parameters\n----------` / `Returns\n-------` blocks to Google-style `Args:` / `Returns:` per CLAUDE.md, and added explicit `float` / `np.ndarray` type annotations on the signature so griffe has nothing to grumble about. Verified end-to-end: rendered HTML pages for `environmental` / `derivedvalues` / `humantohuman` now carry 1 / 7 / 1 `arithmatex` markers respectively, matching the `$$…$$` block count per file. Five `DerivedValues.__call__` math blocks tidied with leading prose ("Spatial hazard per location and tick:", "Prevalence fraction and its time-average per location:", etc.); the two redundant `C_{ij}` formulations joined with "Equivalently:" rather than dropped.
- 🦺 Mark assessment item "polyfill.io in `mkdocs.yml`" as ✅ Fixed. The `<script src="https://polyfill.io/v3/polyfill.min.js?features=es6">` injection has been removed from `mkdocs.yml` (the domain was sold to Funnull in early 2024 and began serving redirect-to-malware payloads to mobile visitors; MathJax 3 needs no polyfill on supported browsers). `grep polyfill mkdocs.yml` now exits non-zero. The Status update section's "Issues that surfaced after the audit" entry and the refreshed-prioritization checklist both mark the item closed.
- 📝 Retract assessment item L8 (`dict_to_propertysetex` "mutates in place"). The function's job is to ingest a raw dict and return a typed `PropertySetEx`, so the attribute reassignment that drove the original concern is the feature, not a side effect — and no real-world caller keeps a parallel reference to the pre-coercion dict. Marked with `~~…~~ — RETRACTED` in the same style as H2 / H5 / W1; the Status update's "Still open" list now strikes L8 through with a pointer to the retraction note.
- 🚚 Move the `Eradication` test-only component out of production code: deleted `src/laser/cholera/test.py` (whose misleading filename had been flagged as assessment item M10) and recreated `Eradication` as `tests/eradication.py`. Updated `tests/test_environmental.py` and `tests/test_envtohuman.py` to `from eradication import Eradication` — resolves through pytest's default `prepend` import mode for `tests/` and via the existing `PYTHONPATH={toxinidir}/tests` under tox. Pytest does not collect the new file as a test module (its `python_files` pattern is `test_*.py` / `*_test.py` / `tests.py`). Closes M10. Also dropped the stale `docs/reference/test.md` (auto-generated stub that pointed at the now-removed module). Full suite (269 tests) still passes; tox check clean.
- 📝 Refresh `assessment.md` with a "Status update — 2026-06-18" section listing what's been closed since the original 2026-06-16 audit (H1-H5, M6, M7, L1-L3, L5) and what remains open (residual H4 sweep, M1-M5, M8-M10, L4, L6-L8, W3-W5), plus issues that surfaced after the audit (polyfill.io supply-chain risk in `mkdocs.yml`, the `--over` partial-window CLI gap, recurring stray `*.log` files) and a refreshed prioritization. Original "Recommended prioritization" / "What I'm not worried about" sections retained verbatim as historical record.
- 🧪 Add `tests/test_logsetup.py` covering `setup_logging` for both string (`"DEBUG"`) and integer (`logging.INFO`) loglevel arguments. The tests save and restore the module-level `_log_file_handler` plus the `laser.cholera` logger's level and handler list in `setUp` / `tearDown` so the import-time invocation does not short-circuit the function-under-test's configuration body, and verify that the `LazyFileHandler` is attached, the logger level is set, and no log file lands on disk before the first emit.
- 🐛 Fix `Eradication.plot()` (src/laser/cholera/test.py) so it is a proper no-op generator. The prior implementation `yield`-ed a bare `None`, which `Model.visualize` would have rendered as a phantom blank PDF page with `plt.title(None)`. Replaced with `yield from ()` (an explicit empty generator) and a docstring describing the protocol it satisfies.
- 📝 Backfill Google-style docstrings across the entire `src/laser/cholera/` tree. Pre-change docstring coverage by AST sweep was 4 / 26 modules, 2 / 22 classes, 19 / 35 free functions, and 8 / 53 methods — most rendered reference pages were 63-character module-header stubs because mkdocstrings' default `show_if_no_docstring: false` hid every undocumented symbol. Added module-level docstrings to all 22 missing modules, class docstrings to all 20 missing classes (every compartment class plus `Component`, `RInterface`, `Model`, `PseEncoder`, `PropertySetEx`, `Parameters`, `Recorder`, `Eradication`), function docstrings to all 16 missing free functions, and method docstrings to all 45 missing methods (component `__init__` / `check` / `__call__` / `plot` quartets plus `Model.plot`, `Parameters.plot`, `Analyzer.plot`, `DerivedValues.plot`, `Recorder.check` / `__call__`). Final AST sweep: `TOTAL UNDOCUMENTED: 0`. Added type annotations along the way to clear every griffe warning: `Iterator[str]` return on each `plot()` generator; `"Model"` (TYPE_CHECKING) forward refs on every component `__init__` / `__call__`; `Optional[Path]` and `**kwargs: object` on `cli_run`; `compute(args: list) -> object`; `setup_logging(loglevel: str, ...) -> None`; `PseEncoder.default`, `PropertySetEx.__init__`, `as_ndarray`, `handle_nan` parameter and return types. Two mkdocs config knobs flipped: `show_if_no_docstring: true` (so undocumented members would still render during the migration), and dropped the redundant `# {identifier}` heading line in `docs/_gen_reference.py` (mkdocstrings already renders the root heading via `show_root_heading: true`). Final build emits zero griffe warnings; previously-empty pages (`metapop/exposed`, `metapop/census`, `metapop/analyzer`, …) now render to ~8 KB of prose each.
- 🦺✨📝🧪 Tighten `metapop` CLI parameter-override validation. `override_helper` (utils.py) now strictly validates `--over` keys against the full `default_parameters.json` schema: scalar keys (`seed`, `phi_*`, `sigma`, `chi_endemic`, `chi_epidemic`, `rho_deaths`, `zeta_ratio`, `delta_reporting_cases`, `delta_reporting_deaths`, `decay_days_spread`, etc. — 26 previously missing entries reconciled) coerce via `int` / `float` / `datetime`; known-but-non-scalar keys (vectors, matrices, `epidemic_peaks` DataFrame, `return` list — every other `default_parameters.json` field) are routed through a new `_cli_unsupported` factory that raises a `ValueError` naming the key and pointing at `--params` / `get_parameters(mods=...)` as the alternatives; unknown keys raise a new `UnknownOverrideKey(ValueError)` subclass with a `difflib`-derived "did you mean" suggestion. In `cli_run` (model.py), `--seed`, `--outdir`, `--params`, and `--loglevel` now have explicit click types (`int`, `click.Path`, `click.Choice`); `--hdf5-output` and `--compress` are promoted to first-class click boolean flags (no more `--over hdf5_output:true`); `override_helper` is now scoped to the parsed `--over` payload only (not the merged click kwargs); `UnknownOverrideKey` is caught and re-raised as `click.UsageError` for a clean CLI message while `_cli_unsupported` `ValueError`s propagate as-is. A `\f` separator keeps the rich Google-style docstring out of `--help`. Drop the obsolete `bool_from_string` / `None`-passthrough test cases in tests/test_metapop_utils.py; add `test_new_scalar_coercions`, `test_unknown_key_raises_with_difflib_suggestion`, `test_unknown_key_with_no_close_match_omits_suggestion`, `test_cli_unsupported_keys_reject_with_helpful_message`. Add CLI-level `TestCliOverrideValidation` to tests/test_metapop.py covering the typo→`UsageError` and architectural-reject→`ValueError` paths. Doctest in `override_helper` exercises the new raise paths.
- 📝🐛 Fix the install-verification example in `docs/installation.md`: the prior `metapop --over date_start:… --over date_stop:… --over nticks:31` smoke test failed with `Shape of b_jt (1155, 40) does not match (nticks, npatches) = (31, 40)` because the CLI does not re-slice the bundled time-series matrices (`b_jt`, `d_jt`, `nu_1_jt`, `nu_2_jt`, `psi_jt`) when the window is narrowed via `--over`, and `nticks` is not in `override_helper`'s coercion table. Replace with a defaults-only `metapop --seed … --loglevel WARNING` invocation (~8 s end-to-end) and add a prose note pointing at the `sim_duration` + matrix-trim pattern in `tests/test_model.py` for programmatic short runs.
- 🚚📝 **Documentation migration from RST/Sphinx to Markdown/MkDocs.** Replace the Sphinx + Furo build with MkDocs + Material + mkdocstrings, served from GitHub Pages via a new `.github/workflows/docs.yml` workflow. All prose pages under `docs/` converted to Markdown (`authors.md`, `installation.md`, `contributing.md`, `changelog.md`, `usage.md`, `index.md`). Top-level `AUTHORS.rst` / `CHANGELOG.rst` / `CONTRIBUTING.rst` renamed to `.md`; `README.rst` deleted (README.md is the canonical readme). Reference docs autogenerated via `docs/_gen_reference.py` + mkdocstrings (replacing `sphinx-apidoc`). Sphinx config (`docs/conf.py`), spell-check wordlist, and `docs/reference/*.rst` stubs all deleted. `pytest.ini`'s `--doctest-glob` flipped from `*.rst` to `*.md`. `pyproject.toml`'s `Documentation` URL repointed to `https://InstituteforDiseaseModeling.github.io/laser-cholera/`. `tox -e docs` now runs `mkdocs build`. See `doc-conversion.md` for the full plan and per-section status.

## 0.10.1 (2026-01-16)

- Add tests for new IFR implementation

## 0.10.0 (2026-01-15)

- New IFR model (Infection Fatality Ratio)
- Update observation process with rho and chi based on infectious prevalence and diagnostic rates
- Update default_parameters.json and LICENSE copyright dates
- Test fixes for NumPy scalar serialization
- Remove MacOS x86_64 from test matrix
- Linter issues and GitHub runner fixes

## 0.9.1 (2025-10-02)

- Fix typo infective -> ineffective
- Add checks against populations going negative
- Expose new_symptomatic
- Only print if verbose is True in parameters
- Skip likelihood check unless "calc_likelihood" is in parameters
- Address linter issues
- Bugfix for parameter constraints (alphas)

## 0.9.0 (2025-08-19)

- Support single location configuration

## 0.8.0 (2025-07-24)

- Spatial hazard computation fix (don't transpose pi_ij in model.results)

## 0.7.11 (2025-07-11)

- Trim and transpose for convenience in MOSAIC

## 0.7.10 (2025-07-10)

- Fix bug in double counting Vxinf
- Fix bug in suitability to decay calculations
- Update default parameters
- Fix indexing for human daily seasonality

## 0.7.9 (2025-06-06)

- Rename beta_env to beta_jt_env
- Rename beta_j_seasonality to beta_jt_human and use directly in spatial hazard calculation
- Update pre-commit
- Fix handling of pi_ij matrix math
- Track vaccine doses delivered
- Births should be Poisson rather than binomial
- Rename estimated to simulated for clarity
- Switch from 'agents' to 'people' terminology
- Fix coupling calculation for denominator == 0

## 0.7.8 (2025-05-16)

- Likelihood cleanup for NaNs and all zeros

## 0.7.7 (2025-05-13)

- Calculate log likelihood at end of simulation

## 0.7.6 (2025-05-13)

- Fix logging setup and np.var() usage

## 0.7.5 (2025-05-13)

- Add Python implementation of R tests for likelihood functions

## 0.7.4 (2025-05-07)

- Fix up reading JSON files back into memory (handle actual NaN vs "NA" or "NaN")
- Record incidence (total and per source)
- Adding likelihood functions
- Adding likelihood function tests
- More consistent variable names

## 0.7.3 (2025-04-30)

- Gate file output on hdf5_output and "return" config parameters
- Clean up console output with logging infrastructure
- Add "quiet" parameter to suppress console progress bar (defaults to False for CLI, True for programmatic interface)
- Update GHA to run tests on push to main
- Support params from R (numeric values come in as doubles, but we need an integer for p)

## 0.7.2 (2025-04-24)

- Support for passing dict to get_parameters()
- Tests for run_model() function
- Additional tests for tracking vital statistics (births, non-disease deaths, disease deaths)

## 0.7.1 (2025-04-24)

- Minor version bump

## 0.7.0 (2025-04-23)

- Initial alpha release
- Support passing parameter dictionary to run_model()
- Fix mapping of environmental suitability (psi_jt) to decay parameter (delta_jt)
- Update default_parameters.json with matrices
- Update parameter loading for matrices
- Clean up plotting and fix seasonality phase
- Handle command line parameter overrides
- Enable parameter overrides correctly
- Allow test parameter sets to skip validation
- Pin numpy, numba, and llvmlite versions
- Remove subpackages
- Update laser-core dependency
- Return model from run_model()
- Require Numba that supports NumPy>=2.0
- Update shedding to environment based on theta_j
- Updated parameters including switch from delta_min/delta_max to decay_days_fast/decay_days_slow
- Use decay_shape_1 and decay_shape_2 to parameterize scipy.stats.beta.cdf
- Add version bump, build, and release GHA
- Metapop implementation work-in-progress commits

## 0.0.0 (2024-09-30)

- First release on PyPI
