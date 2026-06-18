# `laser-cholera` — Red-team assessment

Written 2026-06-16. Findings come from a directed audit of the source tree, docs,
tests, and config — plus first-hand observations from extended work in this
codebase. Each finding is labeled with a confidence tag:

- **(verified)** — I directly inspected the file and reproduced the issue.
- **(agent-reported)** — Surfaced by an Explore agent; I have not independently
  verified the exact line numbers or behaviour. Treat as a starting point, not
  ground truth.
- **(observed)** — I encountered this during real work in this repo over the
  last week of conversations; not a fresh inspection.

---

## Executive summary

The package's *core algorithms* are in solid shape: the likelihood code is
well-translated from R, has good unit-test coverage now (~92 %), and the
metapopulation SEIR machinery does what the modelling docs say it does. But the
**plumbing around the core is fragile**. The biggest risks are:

1. ~~**Latent typo in the CLI override table** that breaks `--over sigma:...`~~
   ✅ **Fixed** — H1.
2. ~~**Assertions used as validation** in the params loader — running with `-O`
   silently disables every parameter sanity check.~~
   ✅ **Fixed in `params.py`** — H4. Other files may have similar patterns;
   not yet swept.
3. **Documentation drift**: two READMEs, ~~a placeholder `Homepage` URL~~
   (✅ H3 fixed), ~~a stub usage page~~ (✅ L5 fixed), stale "Spring
   likelihood" terminology in module docstrings, and compartment
   components with no `__call__` docstrings at all.
4. **Workflow hygiene**: stray `.log` files in the repo root, build
   artefacts (`htmlcov/`, `dist/`, `coverage.xml`) on disk (untracked but
   visible noise); ~~and `*.R` reference files shipped under `src/` rather
   than a sibling reference directory~~ ✅ **L1 fixed**.
5. **Silent numerical guard rails**: the NB likelihood floors negative
   estimates to `1e-10` instead of rejecting them.

Nothing here is catastrophic. Most issues are 5–15 minutes of focused work each.
The catastrophic-feeling thing is the *number* of small issues, which suggests
the project has been growing faster than maintenance has kept up.

**Retraction:** an earlier version of this document claimed the in-window
peak filter had been reverted from the Python port. That claim was wrong —
see the retraction note in finding H5 below.

**Status legend** — findings carry a top-line `**Status:** ✅ Fixed` marker
when they have been addressed in the repo. The original finding text is
preserved unmodified below the status so the audit history is intact.

---

## Critical / High-severity findings

### H1. Typo `"simga"` in `override_helper` mapping table   (verified)

**Status:** ✅ Fixed — the production mapping table now reads `"sigma": float`, and
`tests/test_metapop_utils.py` has `test_all_float_overrides_are_typed` which
exhaustively walks every float-mapped key (sigma included), so a future
re-introduction of the typo would fail loudly.

**File:** `src/laser/cholera/metapop/utils.py:112`

```python
mapping = {
    ...
    "rho": float,
    "simga": float,   # ← should be "sigma"
    "longitude": None,
    ...
}
```

CLI overrides like `--over sigma:0.5` will be forwarded as a string to
`params.sigma`, and the rest of the pipeline expects a float. This has likely
been broken since the override layer was introduced; the fact that no one has
hit it suggests `sigma` is never overridden via CLI in practice — but it also
means *no test catches it*.

**Fix:** one-character rename; add a test under `tests/test_metapop_utils.py`
that exercises every key in the mapping table at least once.

### H2. ~~`partial(datetime.strptime, format="%Y-%m-%d")` raises `TypeError`~~ stale source

**File:** `src/laser/cholera/metapop/utils.py`, the `mapping["date_start"]` /
`mapping["date_stop"]` entries.

`datetime.strptime` is a C function that **rejects keyword arguments**.
Calling `partial(datetime.strptime, format="%Y-%m-%d")("2024-01-01")` raises
`TypeError: strptime() takes no keyword arguments`. This means `--over
date_start:2024-01-01` has been crashing the override layer.

**Fix:** replace with a positional-args wrapper:
```python
def _parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d")  # noqa: DTZ007
```
(I made this exact fix earlier in this conversation; the rebase may or may not
have preserved it — confirm in the working tree.)

### H3. `pyproject.toml` Homepage URL is a placeholder   (verified)

**Status:** ✅ Fixed — `pyproject.toml:48` now reads
`Homepage = "https://github.com/InstituteforDiseaseModeling/laser-cholera"`.

**File:** `pyproject.toml:48`

```toml
Homepage = "https://example.com"
```

`Repository`, `Issues`, `Changelog` are all set correctly. Only `Homepage` is
still `example.com`. This shows on PyPI when the package is published.

**Fix:** point at `https://github.com/InstituteforDiseaseModeling/laser-cholera`
(or readthedocs / docs.idmod.org).

### H4. `assert` used for input validation in `params.py`   (verified, count: 79 occurrences)

**Status:** ✅ Fixed (params.py) — all 80 input-validation asserts in
`dict_to_propertysetex`, `validate_parameters`, and `Parameters.check()` were
converted to `if/raise ValueError(...)`. Two asserts remain in `params.py` and
are intentional: line 154 is an internal post-condition of dispatch logic
inside `handle_nan`, not user-input validation. The `epidemic_threshold`
type-fallthrough also changed from `RuntimeError` to the more precise
`TypeError`. Three tests in `test_params.py` were updated to expect
`ValueError` instead of `AssertionError`. Running with `python -O` no longer
silently disables the validators.

**Out of scope (still pending):** other files in the package may have similar
patterns. A repo-wide pass would be useful but not done in this round.

**Files:** `src/laser/cholera/metapop/params.py` and many others — a
`grep -c 'assert '` across just `params.py` returns **79 hits**.

These cover everything from `assert int(params.p) == params.p` to entire
range checks. Python's `-O` flag (used by some CI environments and most
release-wheel installs) strips assertions. Result: in production any of these
guards become no-ops.

**Fix:** use `if/raise ValueError(...)` for genuine validation; reserve
`assert` for invariant documentation in pure-Python paths where `-O` is never
expected.

### H5. ~~In-window peak filter ported then reverted~~ — RETRACTED

This finding was wrong. The filter **is** present in `main` (commit
`a09d7ac`, tag `v0.13.1`), in `reporting-fixes` HEAD (`6ef3c3d`), and in
`likelihood-updates` (the original-add commit `f8e3b38`). All three call
sites (main precompute + both legacy helpers) are intact.

The error: during an active rebase I ran `grep "date_lo\|date_hi"
calc_model_likelihood.py` against the working tree and got an empty result.
The working tree at that moment contained unresolved conflict markers; the
committed source on every reachable branch still has the filter. Leaving
this entry here as a record of the mistake — and as a reminder that
filesystem state during a rebase is *not* the canonical source.

---

## Medium-severity findings

### M1. Silent floor on negative estimates inside `_calc_log_likelihood_nb`   (verified)

**File:** `src/laser/cholera/calc_model_likelihood.py:138`

```python
est_m = np.maximum(estimated[mask], 1e-10)
```

If `estimated` contains negative values (which is a *precondition violation*),
this silently floors them to `1e-10` and continues. The function's *outer*
validation rejects `est < 0` at the top of `calc_model_likelihood`, but
`_calc_log_likelihood_nb` is also exported and callable directly (it's
underscore-prefixed but still importable). Direct callers get a wrong but
finite answer rather than an error.

**Fix:** add an explicit precondition check inside `_calc_log_likelihood_nb`
too, or document loudly that the caller must validate.

### M2. Component check() doing work it shouldn't   (agent-reported)

**File:** `src/laser/cholera/metapop/census.py:17` — `Census.check()` allegedly
calls `self(self.model, -1)` for initialization.

Using `check()` for side effects (initialization) violates the harness
contract that `check()` is a verification hook. Other components may break if
their `check()` is called before/after a `Census.check()` that has already
mutated state.

**Fix:** move initialization into `__init__` (matches what every other
compartment component does).

### M3. `Census.__init__` not following the lazy-add pattern uniformly   (observed)

**File:** `src/laser/cholera/metapop/vaccinated.py:48`

```python
if not hasattr(self.model.patches, "non_disease_deaths"):
    self.model.patches.add_vector_property("non_disease_deaths", ...)
```

This lazy-add only runs when `model.components` is ordered such that
`Vaccinated` runs before `Susceptible` (which is also where the
non-disease-deaths array gets added). If component order changes, behavior
changes. The pattern repeats in several components and isn't documented.

**Fix:** document the component-order contract, or make `Model.__init__`
allocate all shared arrays before any component sees them.

### M4. Magic numbers everywhere   (verified)

- `±14` peak window in `_calc_peak_*_from_indices` (`calc_model_likelihood.py`)
  — appears in 4 places, never named.
- `1e6` proportional penalty in Poisson zero-prediction path
  (`calc_log_likelihood_distributions.py`) — never named, never explained.
- `0.001` offset in `get_pi_from_lat_long` docstring is shown but **not in the
  code** — code uses `d[i,j]^-gamma` directly. So the docstring is wrong, or
  the code is missing the offset that the docstring documents.
- `min_obs_for_likelihood = 3` is hard-coded inside the main loop of
  `calc_model_likelihood` rather than exposed as a parameter.
- `k_fallback = 10.0` in `ll_cumulative_progressive_nb` is hardcoded; the R
  version reads it from an option.

**Fix:** named module-level constants with docstring explanations. Where the
value is a *modelling choice* (peak window width, NB k floor), expose as a
keyword argument with a sensible default.

### M5. `compute_wis_parametric_row` returns `np.nan` from a function whose other paths return floats   (observed)

**File:** `src/laser/cholera/calc_model_likelihood.py`

The function returns `np.nan` on degenerate inputs and a `float` otherwise.
Downstream code in `calc_model_likelihood` checks `if np.isfinite(wis_c):` —
good — but the function's return type annotation is `-> float`, and the
docstring's `Returns:` says "WIS score (lower is better)" without mentioning
that NaN means "could not compute". Easy to write a new caller that crashes.

**Fix:** annotate the return as `float` with a docstring `Returns:` block that
explicitly enumerates the NaN sentinel and what triggers it.

### M6. Two competing READMEs   (verified)

**Files:** `README.md` (uv-flavoured installation notes) and `README.rst`
(full sphinx-flavoured badges + install).

- `README.md` points at `docs.idmod.org/projects/laser-cholera/`.
- `README.rst` points at `laser-cholera.readthedocs.io/en/latest/`.

These are *different docs sites*. At least one of them is wrong. PyPI shows
whichever is named `README.md` (per `pyproject.toml`'s
`readme = "README.md"`), so `README.rst` is dead weight that confuses
contributors.

**Fix:** delete `README.rst`; consolidate everything into `README.md`; pick
*one* canonical docs URL.

### M7. Compartment components have no `__call__` docstrings   (agent-reported)

**Files:** `src/laser/cholera/metapop/susceptible.py`, `exposed.py`,
`infectious.py`, `recovered.py`, `vaccinated.py`, `census.py`,
`environmental.py`, `envtohuman.py`, `humantohuman.py`.

These are the *per-tick* dispatch functions. A new contributor reading
`exposed.py` has no idea what the function does without staring at the code.
Some are 50+ lines of vectorised NumPy with no comment explaining the
epidemiological intent.

**Fix:** at least a Google-style docstring per `__call__` with a one-sentence
purpose, the math expression it implements, and the side effects on
`model.people` / `model.patches`.

### M8. Tox configured to test only one Python version   (agent-reported)

**File:** `tox.ini` — environments for Python 3.10, 3.11, 3.13, 3.14 are
commented out; only `py312` is active. `pyproject.toml` advertises all five
classifier versions.

**Fix:** either un-comment the environments (and accept the CI runtime) or
update `pyproject.toml` to advertise only what's actually tested.

### M9. Stray `*.log` files at the repo root   (verified)

`20260616111728.log`, `20260616141949.log`, `20260616142119.log`,
`20260616142759.log`, and similar files from earlier sessions are in the
working tree. They're *not* tracked by git (confirmed via `git ls-files`), so
they won't end up in commits — but they clutter `git status` and confuse
new developers.

A pre-commit hook recently added `*.log` to `.gitignore`. Good. But the
existing files should be deleted.

**Fix:** `rm *.log` in the repo root (low risk; nothing references them).

### M10. The "test" module isn't a test   (verified)

**File:** `src/laser/cholera/test.py` — contains an `Eradication` component
class. Not a test. Naming is misleading; if someone runs
`pytest src/laser/cholera/test.py` they'll be confused.

**Fix:** rename to `eradication.py` and move into `metapop/` (which is where
all other compartment-y components live).

---

## Low-severity findings

### L1. R reference files shipped under `src/`   (verified)

**Status:** ✅ Fixed — all 9 `.R` files moved out of `src/laser/cholera/` and
`tests/` into a new repo-root `reference/` directory. The narrow `.gitignore`
rules (`src/laser/cholera/*.R` / `tests/*.R`) were removed so the reference
files are now tracked in git (visible to contributors). The sdist excludes
them by virtue of `reference/` not appearing in
`[tool.uv.build-backend].source-include`; a comment block above that list in
`pyproject.toml` warns future maintainers off adding `reference/*` to it.
Verified by building an sdist and confirming `tar -tzf` shows no `.R` files
or `reference/` directory entries. Five path references in
`docs/likelihood.md` were updated from `tests/...` and `src/laser/cholera/...`
to `reference/...`.

**Files:** `src/laser/cholera/calc_model_likelihood.R` and similar are present
under `src/`. `.gitignore` includes `src/laser/cholera/*.R` to keep them out of
git, but their *presence in the source tree* means they get packaged into
sdists if the build backend isn't carefully configured.

**Fix:** move them to a sibling `reference/` directory at the repo root, or
verify the wheel/sdist build excludes them.

### L2. Test methods without docstrings   (agent-reported)

**Status:** ✅ Fixed — every `test_*` method under `tests/` now has a
Google-style given-when-then docstring with a "failure implies …" note.
52 methods across 14 files updated:
`test_census.py` (3), `test_core.py` (1), `test_environmental.py` (3),
`test_envtohuman.py` (6), `test_exposed.py` (2), `test_humantohuman.py` (3),
`test_ifr.py` (5), `test_infectious.py` (9), `test_metapop.py` (1),
`test_metapop_utils.py` (2), `test_model.py` (5), `test_recovered.py` (3),
`test_susceptible.py` (4), `test_vitalstatistics.py` (5). Confirmed with an
AST walk: zero remaining `test_*` methods lack a docstring. Inconsistency
notes are surfaced inline where the test relies on conventions worth
flagging (assert-True placeholders, name typos like
``test_infectous_steadystate``, non-strict `<=` in stochastic tests, etc.).

Across `tests/test_census.py`, `test_environmental.py`, `test_humantohuman.py`,
`test_exposed.py`, `test_recovered.py`, `test_susceptible.py`,
`test_vitalstatistics.py`, `test_ifr.py`, `test_metapop.py` — at least 25 test
methods have zero docstrings. `CLAUDE.md` explicitly requires given-when-then
docstrings on every test.

**Fix:** mechanical pass adding docstrings; can be done incrementally.

### L3. `assert True` / `# assert True` placeholder tests   (agent-reported)

**Status:** ✅ Fixed — the four `assert True` placeholders in
`tests/test_model.py` (the `test_run_model_None` / `_string` / `_path` /
`_dict` dispatch tests) were replaced with calls to a new module-level
helper `_assert_model_ran(model, expected_nticks, expected_n_patches)`
that pins the real structural invariants of a successful `run_model`
return: non-None object, ``nticks + 1`` rows of per-tick state,
non-negative compartment counts across S/E/Isym/Iasym/R/V1/V2, recorded
`patches.births` / `patches.non_disease_deaths`, and an analyzer-set
`model.log_likelihood` attribute. The `try`/`self.fail(...)` wrappers in
the string and path tests were removed — they hid the failure site;
unwrapped exceptions now produce useful tracebacks. The stale
"Inconsistency note" paragraphs in the L2 docstrings that pointed at the
old placeholders were also trimmed. All 5 tests in `test_model.py` still
pass; full suite (267 tests + 47 subtests) still green.

**Not done:** the four dispatch paths should arguably produce
*numerically identical* models when given the same parameters, which
would require a fixed RNG seed in `run_model`'s defaults. Out of scope
for L3; would make a useful follow-up test (e.g.,
``np.array_equal(model_from_none.people.S, model_from_dict.people.S)``).

**File:** `tests/test_model.py` (lines 46, 60, 74, 84) — and historically
elsewhere. `# assert True` passes whether the body of the test crashed or not.
Real test docstring says one thing; assertion verifies nothing.

**Fix:** convert to real assertions on observable state.

### L4. 10 unattached `TODO` comments   (agent-reported, verified count = 10)

Across `params.py`, `envtohuman.py`, `humantohuman.py`, `environmental.py`,
`recorder.py`, `utils.py`. Examples: `# TODO - is this necessary?`,
`# TODO - TBD`. No ticket references, no completion criteria.

**Fix:** move to GitHub issues with acceptance criteria, or delete.

### L5. `docs/usage.rst` is essentially a stub   (verified earlier)

**Status:** ✅ Fixed — `docs/usage.rst` was rewritten as a real usage
guide with five sections: *Quick start* (CLI invocation via ``metapop``
and Python invocation via ``run_model``), *Overriding parameters* (the
``--over`` CLI flag, the ``mods=`` kwarg, and ``sim_duration`` for short
runs), *Computing the model log-likelihood* (the existing content from
the prior version, retained), *Cross-checking against the upstream R
reference* (testthat setup + ``Rscript -e 'testthat::test_file(...)'``
invocations against the ``reference/`` directory), and *Where to look
next* (cross-references into the autodoc reference). The competing
``docs/likelihood.md`` (which had been a stray Markdown file in an
RST-only Sphinx build with no ``myst-parser`` configured, and was not in
the toctree) was deleted; its content is now consolidated in
``usage.rst``. The doctest example in ``usage.rst`` still passes via
``--doctest-glob=*.rst``.

**Bug found along the way:** the prior version's integration recipe
pointed at ``model.patches.reported_deaths`` for the est_deaths argument.
The actual analyzer (verified in
``src/laser/cholera/metapop/analyzer.py:51``) uses
``model.results.reported_deaths``. Fixed in the rewrite.

**Decision made (per the original "Markdown vs RST" question):** lean
RST. The Sphinx config is already RST-only, all other docs are RST,
adding ``myst-parser`` would be a new dependency for one file's worth of
content. RST is the canonical format going forward.

I expanded it during a prior conversation, then it got rewritten/reverted.
Current state mentions a real function but is still thin: no end-to-end
tutorial, no recipe for the analyzer integration, no notes on the calibration
workflow. The new `docs/likelihood.md` covers some of this, but it's an .md
file in an rst-flavoured docs build that may or may not render.

**Fix:** decide whether to lean Markdown (via `myst-parser`) or RST, and
commit. Right now there's one of each, both partial.

### L6. RST math in Google-style docstrings   (agent-reported)

**Files:** `src/laser/cholera/metapop/humantohuman.py:65`,
`derivedvalues.py:35`. `.. math::` blocks in docstrings the `CLAUDE.md`
conventions say should be Google-style markdown.

**Fix:** convert to MathJax-friendly inline math compatible with the chosen
docs build.

### L7. `docs/requirements.txt` underspecifies   (agent-reported)

`sphinx>=1.3` (from 2015), no version pin, no `furo` (used in `conf.py`), no
`myst-parser` if we go Markdown. `pyproject.toml`'s `[docs]` extra is
commented out.

**Fix:** specify all docs dependencies with reasonable upper bounds; uncomment
and align the `[project.optional-dependencies] docs = [...]` block.

### L8. ~~`dict_to_propertysetex` mutates `params` in place~~ — RETRACTED

This finding does not survive scrutiny. The function's job *is* "take a
raw dict, return a typed `PropertySetEx`." The type coercion via
attribute reassignment isn't a side effect — it's the whole reason the
function exists, and the name already says so (`dict` → `PropertySetEx`,
i.e. an ingest-and-type conversion).

The original entry also flagged "anyone holding a reference to the
original dict will see fields shift type." In practice every caller is
either `load_json_parameters` / `load_compressed_json_parameters`
(one-shot internal use — the dict came out of `json.load` and is
discarded immediately) or `get_parameters(paramsource=dict)` (where
the caller passed the dict *in order to get it back typed*). No
real-world caller keeps a parallel reference to the pre-coercion dict.

Leaving the entry here so the audit history is intact, but marking it
as not actionable.

**File:** `src/laser/cholera/metapop/params.py:167`

The function takes a Python dict, wraps it in a `PropertySetEx`, and then
*re-assigns* attributes on the result to coerce types (`params.date_start =
datetime.strptime(...)`). Anyone holding a reference to the original dict's
fields will see them shift type. This isn't strictly wrong, but it's surprising,
and the function's name says "convert", not "convert and re-type".

**Fix:** make the type coercion explicit in the function name (e.g.,
`ingest_parameters`), or document the side effect.

---

## Workflow / process observations

### W1. ~~The rebase is currently broken~~ — RETRACTED

This finding was an artifact of the moment the assessment was written.
At that moment `git status` reported
``interactive rebase in progress; onto a09d7ac`` with three conflict
regions in ``tests/test_calc_model_likelihood.py``. The rebase has since
been completed (conflicts resolved, ``reporting-fixes`` is now clean and
up-to-date with ``origin/reporting-fixes``) — but the original entry made
it sound like the broken state was a *property* of the repo, which it
was not. It was a transient property of a particular working directory.

Leaving the entry here so the audit history is intact, but treating the
broader "skeptical of current source claims" framing as no longer
applicable: post-rebase, the source state on each branch is what `git`
reports it to be.

### W2. Coverage gaps were hidden by uninstalled tooling
`coverage` and `pytest-cov` were not installed in the project's `.venv` until
this audit. As a result, coverage was being reported but not actually computed
during normal development. After installing, I found `analyzer.py` was at
69 %, `calc_log_likelihood_distributions.py` at 31 %, etc. Most of those are
now fixed but the *gap between assumed coverage and actual coverage* is the
real risk.

### W3. The R reference is the source of truth and it drifts
This package has been ported from R, with R files committed under `src/`. The R
files get upstream commits that the Python doesn't always pick up promptly
(e.g., the in-window peak filter described in H5). There's no automated
diff/check between the R and the Python.

**Suggestion:** a lightweight `tools/sync-with-r.py` that diffs function
signatures and key constants, run as part of CI on schedule.

### W4. The component-execution contract is implicit
Components do not declare their dependencies (which `params` keys they read,
which `people`/`patches` arrays they expect, which they create). The component
order in `model.components = [Susceptible, ..., Vaccinated, ...]` is the only
encoding of the contract. Re-ordering breaks things in non-obvious ways.

**Suggestion:** each component declares `reads = ("params.S_j_initial", ...)`,
`writes = ("people.S", "patches.non_disease_deaths", ...)`, and the model
validates the ordering matches the declared dependencies.

### W5. Numerical sentinels are inconsistent
Across the codebase, "I couldn't compute this" is variously expressed as
`np.nan`, `-np.inf`, `0.0`, `np.inf`, or `None`. Different functions use
different sentinels for the same logical condition (e.g., "insufficient data").
Downstream code has to know which to check.

**Suggestion:** module-level constants `NO_DATA = float("nan")`,
`UNREACHABLE = -np.inf`, etc., with docstrings explaining when each fires.

---

## Status update — 2026-06-18

Two days after the original audit. The original "Recommended prioritization" / "What I'm not worried about" sections below are preserved as historical record; this section is the current state.

### Closed since the audit

| ID | Title | Resolution |
|---|---|---|
| **H1** | `"simga"` typo | mapping now reads `"sigma": float`; `test_metapop_utils.py::test_new_scalar_coercions` exhaustively walks every coercible key including `sigma`. |
| **H2** | `partial(strptime, format=...)` `TypeError` | utils.py uses a positional-args `_parse_date` wrapper. |
| **H3** | `pyproject.toml` Homepage placeholder | now points at the GitHub repo. |
| **H4** | `assert` for validation in `params.py` and the compartment modules | 80 input-validation asserts in `params.py` (`dict_to_propertysetex` / `validate_parameters` / `Parameters.check()`) converted to `if/raise ValueError(...)` in the first pass. Second pass (2026-06-18) extended the conversion to all 10 compartment modules: 105 more converted (49 `hasattr` → `AttributeError`, 47 inline `in` → `ValueError`, 9 multi-line `in` → `ValueError`). The 16 remaining asserts are legitimate invariants (13 post-binomial-draw non-negativity checks; 3 shape-equality checks in constructor / inner-loop code). Every modified `Raises:` docstring block updated to enumerate `AttributeError` / `ValueError` instead of the generic `AssertionError`. |
| **H5** | In-window peak filter (retraction) | retraction confirmed; filter present in all three branches. |
| **M6** | Two competing READMEs | `README.rst` deleted; `README.md` is the only readme. |
| **M7** | Compartment `__call__` docstrings | every compartment module / class / `__init__` / `check` / `__call__` / `plot` now has a Google-style docstring; AST sweep reports 0 undocumented symbols across the whole `src/laser/cholera/` tree. mkdocstrings now also has `show_if_no_docstring: true` so partial coverage during future migrations does not produce blank pages. |
| **L1** | R reference files under `src/` | moved to repo-root `reference/`, sdist excludes them. |
| **L2** | Test methods without docstrings | every `test_*` in `tests/` now has a given-when-then docstring. |
| **L3** | `assert True` placeholders | replaced with structural assertions via `_assert_model_ran` in `tests/test_model.py`. |
| **L5** | `docs/usage.rst` stub | rewritten as a real usage guide; subsequently superseded by the RST→Markdown/MkDocs migration. The current `docs/usage.md` is doctest-verified. |
| **M10** | `src/laser/cholera/test.py` misleading filename | the file has been deleted; `Eradication` (its only contents, a test-only component) moved to `tests/eradication.py`. Imports in `tests/test_environmental.py` and `tests/test_envtohuman.py` updated to `from eradication import Eradication` — resolves through pytest's default `prepend` import mode and via `PYTHONPATH={toxinidir}/tests` under tox. Full suite (269 tests) still passes. |
| **L6** | RST `.. math::` blocks in docstrings | all 8 occurrences (`environmental.py:map_suitability_to_decay`, `humantohuman.py:HumanToHuman.__call__`, and four more in `derivedvalues.py`: `DerivedValues.__call__` × 5 blocks plus `calculate_spatial_hazard` and `calculate_coupling`) converted to `$$…$$` arithmatex/MathJax syntax. Affected docstrings flipped to `r"""…"""` so single-backslash LaTeX reads cleanly in source. Numpydoc-style `Parameters` / `Returns` blocks in `map_suitability_to_decay` also rewritten as Google-style `Args:` / `Returns:` with explicit type annotations on the function signature. |
| **L7** | `docs/requirements.txt` drift from `[docs]` extra | `docs/requirements.txt` deleted; `pyproject.toml`'s `[docs]` extra is now the single source of truth. `tox.ini`'s `[testenv:docs]` switched to `extras = docs`. (`.readthedocs.yml` itself was deleted as part of the full RTD removal — see next entry.) |
| **RTD removal** | The repo was still partly wired up for Read the Docs even though docs are served from GitHub Pages | `.readthedocs.yml` deleted; `docs/index.md`'s "migration in progress" / "render on Read the Docs" paragraph replaced with a description of the actual MkDocs/Material/GitHub-Pages pipeline; the stale `mkdocs.yml` comment about "Prose pages will be added to `nav` as they migrate from .rst to .md" trimmed; the `[ ] Decide whether to keep the RTD project` task in `doc-conversion.md §8` marked done. References to RTD in `CHANGELOG.md`, `assessment.md`, and `doc-conversion.md` left in place as historical context. |

### Still open

**High-severity follow-ups**

- ~~**H4 (residual)** — the repo-wide sweep of `assert` used for validation has only landed in `params.py`.~~ ✅ **Fixed** — applied the same `assert` → `if/raise` conversion to every compartment module. 105 validation asserts converted across the 10 metapop component files: 49 `assert hasattr(X, "Y"), msg` → `if not hasattr(X, "Y"): raise AttributeError(msg)`, 47 inline `assert "X" in Y, msg` → `if "X" not in Y: raise ValueError(msg)`, 9 multi-line (line-wrapped) variants of the same. The 16 asserts that remain are the kinds the original audit said should stay: 13 post-binomial-draw `assert np.all(X >= 0)` non-negativity invariants and 3 internal shape-equality checks in `envtohuman.__init__` / `calculate_coupling`. Each modified `Raises:` docstring block was rewritten to enumerate `AttributeError` / `ValueError` instead of the generic `AssertionError`. Full test suite (269) still passes; lint clean.

**Medium-severity**

- **M1** — `_calc_log_likelihood_nb` still floors negatives to `1e-10`. No explicit precondition check in the underscore-prefixed entry point.
- **M2** — `Census.check()` still calls `self(self.model, -1)` to seed `N[0]`. The new docstring acknowledges the hack but doesn't move it.
- **M3** — lazy-add pattern in `Vaccinated` / `Susceptible` / others is still implicit; component-order contract still undocumented.
- **M4** — magic numbers still un-named (`±14` peak window, `1e6` Poisson penalty, `min_obs_for_likelihood = 3`, `k_fallback = 10.0`). The `0.001` docstring / code mismatch in `get_pi_from_lat_long` is also still there (docstring claims a `+ 0.001` distance offset; code uses `d^-gamma` directly with no offset).
- **M5** — `compute_wis_parametric_row` still returns `np.nan` from a function annotated `-> float`; no sentinel documentation in the `Returns:` block.
- **M8** — `tox.ini` still tests only `py312`; the `py310` / `py311` / `py313` / `py314` envs are still commented out despite the `pyproject.toml` classifier list advertising them.
- **M9** — fresh `*.log` files keep accumulating at the repo root (`20260616174701.log`, `20260617114833.log`, etc.). `.gitignore` covers them so commits are clean, but the working-tree clutter remains.
- ~~**M10** — `src/laser/cholera/test.py` still exists as a misleading filename.~~ ✅ Closed; see the table above.

**Low-severity**

- **L4** — 10 unattached `TODO` comments still in the tree (`params.py`, `envtohuman.py`, `humantohuman.py`, `environmental.py`, `recorder.py`, `utils.py`).
- ~~**L6** — `.. math::` RST blocks still present in `environmental.py:193`, `humantohuman.py:121`, and `derivedvalues.py:96/100/104/108/110/113`.~~ ✅ **Fixed** — all eight `.. math::` blocks (the originally-flagged six plus two more found during the sweep, in `calculate_spatial_hazard` and `calculate_coupling`) converted to `$$…$$` Material/arithmatex syntax. Affected docstrings switched to `r"""…"""` so the single-backslash LaTeX source is readable in the file as well as in the rendered HTML. While in there, the numpydoc-style `Parameters\n----------` / `Returns\n-------` blocks in `map_suitability_to_decay` were rewritten as Google-style `Args:` / `Returns:` for `CLAUDE.md` consistency, and explicit type annotations were added to that function's signature. Rendered HTML pages for the three reference modules now carry the expected arithmatex `<span class="arithmatex">` markers (1 / 7 / 1 blocks across `environmental`, `derivedvalues`, `humantohuman`); MathJax picks them up.
- ~~**L7** — `docs/requirements.txt` and `pyproject.toml`'s `[docs]` extra are still partially duplicated.~~ ✅ **Fixed** — `docs/requirements.txt` deleted; the `[docs]` extra in `pyproject.toml` is now the single source of truth for the docs toolchain. `tox.ini`'s `[testenv:docs]` swapped from `deps = -r{toxinidir}/docs/requirements.txt` to `extras = docs` (canonical tox idiom). `.readthedocs.yml` swapped from `requirements: docs/requirements.txt` to `pip` install with `extra_requirements: [docs]`. GitHub Actions `docs.yml` was already using `uv run --extra docs …` so no change there. **NB**: `.readthedocs.yml` still references `docs/conf.py` (which was deleted in the Sphinx→MkDocs migration), so the file is otherwise stale — RTD cannot currently build the project from this config regardless of how requirements are sourced. Out of scope for L7 — see "Issues that surfaced after the audit" below for the broader RTD-vs-GitHub-Pages decision still pending.
- ~~**L8** — `dict_to_propertysetex` still mutates the wrapped `PropertySetEx` in place during type coercion.~~ Retracted; the in-place type coercion is the function's purpose, not a surprise. See the retraction note in the L8 entry above.

**Workflow**

- **W3** — no automated diff between the R reference (`reference/`) and the Python port. Not started.
- **W4** — component dependencies still implicit. Not started.
- **W5** — sentinel inconsistency (`np.nan` / `-np.inf` / `0.0` / `None`) still unresolved.

### Issues that surfaced after the audit

- **Security: `polyfill.io` in `mkdocs.yml`.** ✅ **Fixed** — the `<script src="https://polyfill.io/v3/polyfill.min.js?features=es6">` injection on `mkdocs.yml:85` has been removed; a `grep polyfill mkdocs.yml` now exits non-zero. The `polyfill.io` domain was sold to Funnull in early 2024 and started serving redirect-to-malware payloads; MathJax 3 doesn't need a polyfill on any browser we care about, so removal had no functional cost.
- **CLI override gap (closed alongside install-docs fix).** The `installation.md` short-window smoke test was broken because `--over nticks:31` would forward as a string and the CLI doesn't re-slice the bundled time-series matrices when the window is narrowed. Closed by (a) fixing the installation example to use the defaults-only invocation and (b) tightening `override_helper` to strict-validate every `--over` key against `default_parameters.json` (unknown keys raise `UnknownOverrideKey` with a difflib suggestion; non-scalar keys raise `ValueError` pointing at `--params`). The underlying CLI gap — partial-window overrides don't truncate the matrices — is still architectural; no plan to fix.
- **Stale `*.log` files** — recurrence of M9. Worth attaching a `clean-logs` make target.
- ~~**`.readthedocs.yml` is stale.**~~ ✅ Resolved — RTD removed entirely. `.readthedocs.yml` deleted; `docs/index.md` and `mkdocs.yml` cleaned of stale RTD / RST-migration prose. See the "RTD removal" entry in the closed table above. Any external links pointing at `laser-cholera.readthedocs.io` should be updated to the GitHub Pages URL as they're encountered.

### Refreshed prioritization

If you have ~half a day:

1. ~~**Delete the polyfill.io script tag** from `mkdocs.yml` (security). 30 seconds.~~ ✅ Done.
2. **M9** (rm \*.log) plus a `clean` target so it stays clean. 5 minutes.
3. ~~**L6** (RST math → Material-compatible inline math). 15 minutes; 3 files.~~ ✅ Done — also picked up the numpydoc → Google-style conversion in `map_suitability_to_decay` and two extra `.. math::` blocks not in the original L6 file list.
4. ~~**M10** (rename `src/laser/cholera/test.py` → `eradication.py`; find usages). 10 minutes.~~ ✅ Done — moved to `tests/eradication.py`.

If you have ~two days:

5. ~~**H4 residual** — repo-wide `assert`-as-validation sweep beyond `params.py`.~~ ✅ Done — 105 validation asserts converted across the 10 compartment modules; the 16 remaining asserts are invariant / shape checks.
6. **M1** — explicit precondition check inside `_calc_log_likelihood_nb`.
7. **M4** — name the magic numbers; reconcile the `0.001` docstring/code mismatch in `get_pi_from_lat_long`.
8. **L4** — convert the 10 TODO comments to GitHub issues with acceptance criteria, or delete them.

If you have ~a week:

9. **W4** — declared component dependencies + ordering validation.
10. **W3** — automated R-vs-Python diff in CI.
11. **M8** — un-comment the `py310` / `py311` / `py313` / `py314` tox envs (or trim `pyproject.toml`'s classifier list).

---

## Recommended prioritization

If you have ~half a day:

1. **H1 + H2** (utils.py typos / strptime) — 15 minutes. Adds tests.
2. **H3** (pyproject Homepage URL) — 2 minutes.
3. **H5** (re-apply in-window peak filter) — 30 minutes. The diff is the same
   as the upstream R commit; reconstruct from this conversation's history or
   from the R source.
4. **M6** (delete `README.rst`) — 5 minutes.
5. **M9** (delete stray `.log` files) — 30 seconds.
6. **M10** (rename `test.py` → `eradication.py`) — 5 minutes (find usages,
   update imports).

If you have ~two days:

7. **H4** (audit and replace `assert` with `ValueError` in `params.py`)
8. **M7** (add `__call__` docstrings to every compartment component)
9. **L2** (mechanical docstring pass on the unscoped test files)
10. ~~**W1** (finish the rebase cleanly).~~ — retracted; the rebase was
    already finished before the audit was committed.

If you have ~a week:

11. Pull the rest of the calc_log_likelihood_* test files up to parity
    coverage with the negbin one (the verbose paths are there but the
    structural assertions could be deeper).
12. Move R files out of `src/`.
13. Establish a sync-with-R workflow (W3).
14. Replace the implicit component-execution contract with declared
    dependencies (W4).

---

## What I'm *not* worried about

- The core likelihood implementation. It's well-translated, well-tested, and
  matches the R numerical behaviour in the cases I've checked.
- The metapop SEIR step functions. They look correct after the recent rho/chi
  rework; the unit tests for the dose-clamping branches confirm the edge cases
  behave.
- Performance. There are some suboptimal patterns (per-tick DataFrame
  filtering would dominate at scale) but for the current use case the runtime
  is dominated by simulation, not by likelihood scoring.
- Security. There's no untrusted input path; the JSON params loader is the
  closest thing and it's only consumed in-process.

The risk profile of this codebase is "quietly wrong" rather than "loudly
broken." That's the kind that bites a year into a calibration study, when
someone realises the LL has been off by 3 % because of an out-of-window peak
date getting clamped to t=0. Hence the emphasis above on fixing silent
guard-rails, not on chasing dramatic performance wins.
