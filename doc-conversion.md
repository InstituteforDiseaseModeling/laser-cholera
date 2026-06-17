# Plan: convert docs from RST/Sphinx to Markdown/MkDocs

**Owner:** TBD &nbsp;·&nbsp; **Status:** Not started &nbsp;·&nbsp; **Drafted:** 2026-06-16

Goal: replace the current Sphinx + RST build with MkDocs + Material + mkdocstrings,
serve the built site from GitHub Pages via a GitHub Actions workflow, and retire
the Read the Docs hosting once the new site is verified.

Track progress by checking the boxes in each section. Sections are roughly
ordered for execution; ones marked **(parallel-safe)** can be done out of order.

---

## 0. Inventory — what we have today

Pre-conversion snapshot, captured 2026-06-16. Re-verify before starting:

- `docs/` contains **6 prose RST pages** (`authors`, `changelog`, `contributing`,
  `index`, `installation`, `readme`, `usage`) plus 4 RST files under
  `docs/reference/` driven by `sphinx-apidoc`.
- `docs/conf.py` enables Sphinx extensions: `autodoc`, `autosummary`,
  `coverage`, `doctest`, `extlinks`, `ifconfig`, `mathjax`, `napoleon`, `todo`,
  `viewcode`. Theme: `furo`.
- `docs/requirements.txt` is minimal (`sphinx>=1.3`, `furo`) — under-specified
  and will need replacement.
- Doctests are wired through pytest via `pytest.ini` (`--doctest-modules`,
  `--doctest-glob=*.rst`). Currently 1 doctest in `docs/usage.rst` and several
  in module docstrings.
- `tox -e docs` runs `sphinx-apidoc` + `sphinx-build -b doctest` +
  `sphinx-build -b html` + `sphinx-build -b linkcheck`.
- `pyproject.toml` advertises
  `Documentation = "https://laser-cholera.readthedocs.io/en/latest/"`.
- 57 instances of Sphinx-specific constructs (`:func:`, `:class:`, `.. automodule::`,
  `.. code-block::`, `.. toctree::`) across the docs that need translation.

---

## 1. Tooling decisions

Pin these up-front so the conversion isn't a moving target.

- [x] **MkDocs core** — current LTS at conversion time. *(resolved to `mkdocs==1.6.x`)*
- [x] **Theme:** Material for MkDocs (`mkdocs-material`). *(resolved to `9.7.x`)*
- [x] **Autodoc replacement:** `mkdocstrings[python]` with the `griffe`
  handler. *(resolved to `mkdocstrings==1.0.x` + `mkdocstrings-python==2.0.x`)*
- [x] **Math:** `pymdownx.arithmatex` + MathJax (via theme config).
  *(via `pymdown-extensions==10.21.x`)*
- [x] **External link shortcuts** for `:issue:` / `:pr:`: decision
  pending — defer the plugin install until §3 needs it. Hand-rewriting
  remains the leading option.
- [x] **API page generation:** `mkdocs-gen-files` + `mkdocs-literate-nav`
  *(resolved to `0.6.x` + `0.6.x`)*.
- [x] **Search:** Material's bundled search.

Pin versions in a new `docs/requirements.txt`:

```text
properdocs>=1.6
mkdocs-material>=9.5
mkdocstrings[python]>=0.27
mkdocs-gen-files>=0.5
mkdocs-literate-nav>=0.6
pymdown-extensions>=10.11
```

(Also expose as a `[project.optional-dependencies]` `docs` extra in
`pyproject.toml` so `uv pip install -e .[docs]` works.)

---

## 2. Add MkDocs alongside Sphinx, do not delete yet

The conversion strategy is **parallel-build** until the MkDocs site is verified.
Don't remove the Sphinx config until the new site is live and reviewed.

- [x] Create `mkdocs.yml` at the repo root (NOT under `docs/`):
  ```yaml
  site_name: LASER Cholera (LASIK)
  site_url: https://InstituteforDiseaseModeling.github.io/laser-cholera/
  repo_url: https://github.com/InstituteforDiseaseModeling/laser-cholera
  repo_name: laser-cholera
  edit_uri: edit/main/docs/

  theme:
    name: material
    features:
      - navigation.sections
      - navigation.expand
      - navigation.top
      - search.highlight
      - content.code.copy
      - content.code.annotate
    palette:
      - scheme: default
        primary: blue
        toggle: { icon: material/brightness-7, name: Switch to dark mode }
      - scheme: slate
        primary: blue
        toggle: { icon: material/brightness-4, name: Switch to light mode }

  plugins:
    - search
    - gen-files:
        scripts:
          - docs/_gen_reference.py
    - literate-nav:
        nav_file: SUMMARY.md
    - mkdocstrings:
        handlers:
          python:
            options:
              docstring_style: google
              show_source: true
              show_signature_annotations: true
              members_order: source
              separate_signature: true

  markdown_extensions:
    - admonition
    - attr_list
    - md_in_html
    - tables
    - toc:
        permalink: true
    - pymdownx.details
    - pymdownx.highlight:
        anchor_linenums: true
    - pymdownx.inlinehilite
    - pymdownx.snippets
    - pymdownx.superfences
    - pymdownx.arithmatex:
        generic: true
    - pymdownx.tabbed:
        alternate_style: true

  extra_javascript:
    - https://polyfill.io/v3/polyfill.min.js?features=es6
    - https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js

  nav:
    - Home: index.md
    - Installation: installation.md
    - Usage: usage.md
    - Contributing: contributing.md
    - Authors: authors.md
    - Changelog: changelog.md
    - API reference: reference/
  ```

- [x] Add `docs/_gen_reference.py` — generates one Markdown stub per Python
  module so `mkdocstrings` can render them. Skeleton:
  ```python
  """Generate one reference page per module under docs/reference/."""
  from pathlib import Path
  import mkdocs_gen_files

  src_root = Path("src/laser/cholera")
  reference_root = Path("reference")
  nav = mkdocs_gen_files.Nav()

  for path in sorted(src_root.rglob("*.py")):
      module_path = path.relative_to("src").with_suffix("")
      doc_path = reference_root / path.relative_to(src_root).with_suffix(".md")
      parts = tuple(module_path.parts)
      if parts[-1] == "__init__":
          parts = parts[:-1]
          doc_path = doc_path.with_name("index.md")
      identifier = ".".join(parts)
      with mkdocs_gen_files.open(doc_path, "w") as fd:
          fd.write(f"# `{identifier}`\n\n::: {identifier}\n")
      nav[parts] = doc_path.as_posix()

  with mkdocs_gen_files.open(reference_root / "SUMMARY.md", "w") as fd:
      fd.writelines(nav.build_literate_nav())
  ```

- [x] Verify the MkDocs build runs cleanly *before* touching any of the
  existing RST files. Build succeeds without `--strict`. There are 7
  remaining warnings, all source-quality (griffe complaining about
  missing type annotations on `date_start` / `date_stop` /
  `epidemic_peaks` in `calc_model_likelihood.py` and a missing return
  annotation in `likelihood.py:603`). These should be fixed as a
  pre-`--strict` cleanup pass; tracked here as **known issue**, not
  blocking the conversion.

---

## 3. Convert prose pages, one at a time

For each RST page below, do: (a) convert to `.md`, (b) preview locally,
(c) link from `mkdocs.yml` nav (already done above), (d) delete the `.rst`.

The order below puts the simplest pages first so the conversion pattern
stabilises before the harder ones.

- [x] **`docs/index.rst` → `docs/index.md`** — short TOC stub. Mostly a copy
  of `README.md`'s headline; the nav itself moves into `mkdocs.yml`.
- [x] **`docs/authors.rst` → `docs/authors.md`** — inlined content (will
  switch to a snippet include when §4 renames `AUTHORS.rst` → `AUTHORS.md`).
- [x] **`docs/installation.rst` → `docs/installation.md`** — expanded
  beyond the one-line stub (closes the second half of L7 in `assessment.md`).
- [x] **`docs/contributing.rst` → `docs/contributing.md`** — inlined the
  CONTRIBUTING content; updated to point at `CHANGELOG.md` / `AUTHORS.md`
  in anticipation of §4 rename.
- [x] **`docs/changelog.rst` → `docs/changelog.md`** — placeholder pointing
  at the GitHub-hosted changelog; will become a snippet include of
  `CHANGELOG.md` after §4.
- [x] **`docs/readme.rst` → ~~`docs/readme.md`~~** — page dropped; `index.md`
  now serves as the homepage and the canonical README is `README.md` at the
  repo root.
- [x] **`docs/usage.rst` → `docs/usage.md`** — full conversion (~200 lines).
  The runnable doctest survived the move and is exercised once §6 flips the
  pytest glob.

**Known issue (not blocking):** the mkdocstrings cross-ref syntax
`[run_model][laser.cholera.metapop.model.run_model]` emits "Could not find
cross-reference target" warnings during build. The links degrade gracefully
to plain code spans in the rendered HTML, so the site is functional, but
the warnings need fixing before `mkdocs build --strict` can ship in CI.
Likely fix: a mkdocstrings option (`show_root_full_path` or similar) or a
change to plugin ordering. Track here; fix in the `--strict` cleanup pass.

### Conversion mechanics

For the bulk-prose pages, a first pass with `pandoc -f rst -t markdown` gets
you 80 % there. Hand-finish:
1. Replace Sphinx code blocks `.. code-block:: python` with fenced ```` ```python ````.
2. Replace cross-refs (`:func:`, `:class:`, `:doc:`, `:ref:`, `:mod:`) with
   either mkdocstrings cross-refs or plain Markdown links.
3. Replace `.. note::` / `.. warning::` admonitions with Material syntax:
   ```
   !!! note
       The body of the note goes here.
   ```
4. Replace `==========` / `----------` underline headings with Markdown
   `# ` / `## `.
5. Replace `\`single backticks\`` with double backticks where the original
   used RST inline literal `` ``foo`` `` (pandoc usually handles this).

---

## 4. Decide what to do with the top-level `*.rst` files

`CHANGELOG.rst`, `AUTHORS.rst`, `CONTRIBUTING.rst`, and `README.rst` are at
the repo root, separate from `docs/`. Some are referenced by `pyproject.toml`
and various tooling.

- [x] **`CHANGELOG.rst` → `CHANGELOG.md`** — mechanically converted via a
  one-shot Python script targeting the actual patterns in the file
  (headings, bullet lists, double-backtick code spans). 143 lines.
  `docs/changelog.md` now uses a `--8<-- "CHANGELOG.md"` snippet include.
- [x] **`AUTHORS.rst` → `AUTHORS.md`** — 6 lines, hand-written. Snippet
  include in `docs/authors.md`.
- [x] **`CONTRIBUTING.rst` → `CONTRIBUTING.md`** — content reused from the
  conversion I'd already done into `docs/contributing.md` during §3.
  Snippet include in `docs/contributing.md`.
- [x] **`README.rst`** — deleted. README.md was already the canonical
  readme; this closes M6 of `assessment.md`.

`pyproject.toml` references update:

- [x] `source-include` updated: `*.md` filenames replacing `*.rst`;
  `README.rst` dropped from the list (since `README.md` is picked up by
  `readme = "README.md"`).
- [x] Two `[[tool.bumpversion.files]]` blocks pointing at `README.rst`
  removed. (The remaining bumpversion entry for `docs/conf.py` stays
  until §5 removes the Sphinx config.)

---

## 5. Reference (autodoc) pages

The current `docs/reference/*.rst` files are auto-generated by `sphinx-apidoc`
each build. With mkdocstrings + `mkdocs-gen-files`, the equivalent is the
`docs/_gen_reference.py` script (skeleton in §2). This means:

- [x] Delete the existing `docs/reference/*.rst` files. Done.
- [x] Verify the generated reference pages render correctly. Build emits
  one page per module under `site/reference/`; the index page links to
  the full per-module list (`calc_model_likelihood/`, `cli/`, `core/`, …,
  and the `metapop/` subdir).
- [x] **Pain point 1 — namespace packages**: handled cleanly by
  `paths: [src]` in mkdocs.yml's mkdocstrings handler config. No
  additional `allow_inspection: false` needed.
- [ ] **Pain point 2 — legacy "Spring likelihood" / "Translation complete"
  narrative** in `calc_model_likelihood.py`'s module docstring. Still
  present and mkdocstrings is dutifully rendering it. This is a *source*
  cleanup, not a docs config issue. Track here; address in the
  pre-`--strict` cleanup pass.

---

## 6. Doctest preservation

The current Sphinx build runs `sphinx-build -b doctest`, and pytest also runs
`--doctest-glob=*.rst`. Both go away with MkDocs (it has no doctest builder).

- [x] Update `pytest.ini`: change `--doctest-glob=\*.rst` to
  `--doctest-glob=\*.md` so the runnable example in `docs/usage.md`
  continues to execute under `pytest`. *(Done as part of §3's commit
  because deleting `usage.rst` without flipping the glob would have
  silently stopped exercising the doctest. The remaining `tox -e docs`
  cleanup stays in §6.)*
- [x] Remove the doctest line from `tox -e docs`'s commands. The whole
  `[testenv:docs]` body is now `mkdocs build {posargs}` — replacing the
  four-line Sphinx incantation (`sphinx-apidoc`, `-b doctest`, `-b html`,
  `-b linkcheck`). Doctests run via pytest, not the docs env.
- [x] Add a `tox -e docs` that just runs `mkdocs build`. The `--strict`
  flag is *not* added yet because of the known cross-ref warnings; flip
  to `--strict` once those are addressed (see §11 done definition).

---

## 7. GitHub Actions workflow for gh-pages

Create `.github/workflows/docs.yml`:

```yaml
name: docs

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0   # needed by mkdocs-material's git-revision-date

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.12

      - name: Install docs extras
        run: |
          uv venv
          uv pip install -e '.[docs]'

      - name: Build with strict warnings
        run: uvx properdocs build -f mkdocs.yml --strict

      - name: Upload Pages artifact
        if: github.event_name == 'push' && github.ref == 'refs/heads/main'
        uses: actions/upload-pages-artifact@v3
        with:
          path: site

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: build
    runs-on: ubuntu-latest
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy to GitHub Pages
        id: deployment
        uses: actions/deploy-pages@v4
```

Choices made above (revisit if you disagree):

- **`actions/deploy-pages@v4`** rather than the older
  `mkdocs gh-deploy` approach. The Pages artifact + deploy-pages pattern is
  GitHub's current recommendation; it doesn't require a `gh-pages` branch
  and works cleanly with branch-protected `main`.
- **PRs build but don't deploy** — catches breakage early without exposing
  PR previews on the public site. Add a PR preview later if useful
  (see "Open questions" §10).
- **`--strict` deliberately NOT used yet.** The known mkdocstrings
  cross-ref warnings would fail every CI run. Switch to `--strict` once
  those are resolved (tracked in §11 done definition).

[x] Workflow file committed at `.github/workflows/docs.yml`.

Repo settings to flip after the first successful deploy (manual, by an
admin — cannot be automated):

- [ ] In **Settings → Pages**, set **Source = "GitHub Actions"** (not
  "Branch"). This activates the deploy-pages action's target.
- [ ] If the repo previously used a `gh-pages` branch for something else,
  decide whether to keep or delete it.

---

## 8. Update external references

After the new site is live at
`https://InstituteforDiseaseModeling.github.io/laser-cholera/`:

- [x] `pyproject.toml`: `Documentation` URL repointed to the GitHub Pages
  URL.
- [x] `README.md`: documentation badge updated to point at the new URL.
- [ ] **Decide whether to keep the RTD project.** Manual decision; not
  automatable. Recommendation: redirect + 2-week notice, then delete.

---

## 9. Cutover order

The point of doing this in-order is to never have the site broken on
`main`. Suggested sequence:

1. §1, §2 — bring up MkDocs alongside Sphinx. Verify local build. No commit
   yet.
2. §3 — convert prose pages. Commit when all convert and the local
   `properdocs serve` looks right.
3. §4 — rename top-level `*.rst` files to `*.md`. Commit separately so the
   rename is reviewable.
4. §5 — wire up auto-generated reference. Commit.
5. §6 — flip doctest glob in `pytest.ini`. Run the full suite to confirm
   no doctest breakage. Commit.
6. §7 — add the GitHub Actions workflow. PR; merge once green.
7. §8 — flip the `pyproject.toml` URL, delete `docs/conf.py`, delete the
   `docs/reference/*.rst` files, delete `docs/requirements.txt`'s old
   contents (replace with the MkDocs ones from §1), drop the `[testenv:docs]`
   block in `tox.ini` (or replace its body with `properdocs build -f mkdocs.yml --strict`).
   Final commit.

If anything goes wrong after step 6 (the workflow is on `main` but the new
site has issues), the rollback is: revert the merge commit. The Sphinx
config in steps 1–5 is still on disk and functional — until step 7 deletes
it.

---

## 10. Open questions

- [ ] **PR previews.** Material + a Cloudflare Pages / Netlify / GitHub
  preview deployment can render each PR's docs at a unique URL. Worth doing
  if reviewers want to eyeball doc PRs. Adds complexity; defer until
  someone asks.
- [ ] **Versioned docs (`mike`).** RTD does this for free; on GitHub Pages
  you need [`mike`](https://github.com/jimporter/mike) to publish multiple
  versions under `/v0.13/`, `/latest/`, etc. We currently only publish
  `latest` on RTD, so we may not need this — but if releases start linking
  back to versioned docs, this becomes load-bearing.
- [ ] **MathJax vs KaTeX.** Material supports both via `pymdownx.arithmatex`.
  MathJax is more featureful but heavier. KaTeX renders faster. The two
  RST math blocks in the codebase use basic syntax that either handles.
  Default to MathJax for parity with the current Sphinx setup.
- [ ] **Should `index.md` duplicate `README.md` or include it via snippet?**
  A snippet keeps them in sync. Duplication keeps the site's homepage from
  being constrained by README conventions. Lean toward snippet for now.
- [ ] **Spellcheck.** `docs/spelling_wordlist.txt` is a Sphinx-specific
  artefact (used by `sphinxcontrib-spelling` if it had been wired up — it
  isn't). Delete during cleanup unless we want to add a
  `mkdocs-spellcheck` plugin equivalent.

---

## 11. Done definition

The conversion is finished when all of the following are true:

- [ ] `properdocs build -f mkdocs.yml --strict` is green locally and in CI. *(Still
  blocked on the known mkdocstrings cross-ref + griffe annotation
  warnings — needs a source-side cleanup pass before flipping.)*
- [ ] The site is live at the new URL. *(Pending the manual repo
  setting "Settings → Pages → Source = GitHub Actions" after this
  branch merges.)*
- [x] `pyproject.toml`'s `Documentation` link points at the new URL.
- [x] `tox -e docs` (if retained) builds via `properdocs` (MkDocs), not Sphinx.
- [x] `pytest tests/ docs/` passes — including the surviving doctest
  in `docs/usage.md`. (199 passing locally as of §8 commit.)
- [x] No remaining `.rst` files under `docs/`. Confirmed by `ls`.
- [x] `docs/conf.py` deleted.
- [x] CHANGELOG entry added describing the migration.

---

## 12. Next steps

Open items as of the §8 commit on the `doc-conversion` branch. Bucketed
by when each item can be tackled.

### Immediate (this PR)

- [ ] **Push the branch and open a PR against `main`.** The CI workflow
  in `.github/workflows/docs.yml` triggers on the PR — it builds the
  MkDocs site but does not deploy. Confirms the workflow runs on real
  Actions runners (so far only verified locally).
- [ ] **Review the rendered output yourself before merging.** Run
  `properdocs serve` locally and eyeball:
  - The `CHANGELOG.md` mechanical conversion — re-read headings and
    code spans to catch anything the script mangled.
  - A couple of compartment-component reference pages
    (e.g., `site/reference/metapop/vaccinated/`) for mkdocstrings
    output sanity.
  - The snippet-include pages
    (`site/authors/`, `site/contributing/`, `site/changelog/`) — confirm
    they render the root-level files, not raw `--8<--` literals.
- [ ] **Merge to `main`.**

### Post-merge (manual, one-time)

- [ ] **Flip the GitHub Pages source.** In the repo's GitHub
  *Settings → Pages*, change *Source* from "Branch" to
  "GitHub Actions". Cannot be automated. After this, the next push to
  `main` triggers the deploy job in `docs.yml`, and the site appears at
  `https://InstituteforDiseaseModeling.github.io/laser-cholera/`.
- [ ] **Verify the first deploy.** The workflow's Actions run will link
  to the resolved Pages URL. Click through; confirm the index page,
  installation page, usage page, and a couple of reference pages
  resolve. If yes, the §11 "site is live" checkbox closes.
- [ ] **Decide on Read the Docs.** Options:
  - (a) delete the RTD project outright,
  - (b) leave it and add a redirect notice to its `index.rst`,
  - (c) **recommended** — redirect notice + 2-week grace period, then
    delete.

### Pre-`--strict` cleanup pass (separate PR, when ready)

The `properdocs build -f mkdocs.yml --strict` flag is not yet in the workflow because three
classes of warnings would fail every CI run. Each needs a separate fix.
Group these into one PR (`docs: pre-strict cleanup`) so the `--strict`
flip lands atomically with the warning fixes.

- [ ] **mkdocstrings cross-ref resolution** in `docs/usage.md`. Five
  `[run_model][laser.cholera.metapop.model.run_model]`-style links fail
  to resolve at build time. Try in order:
  1. Re-verify the `plugins:` ordering in `mkdocs.yml` puts
     `mkdocstrings` *after* `gen-files` and `literate-nav`.
  2. Add `show_root_full_path: true` and/or `show_root_toc_entry: true`
     to the mkdocstrings handler options.
  3. As a last resort, replace the autorefs syntax with explicit
     anchors like
     `[run_model](reference/metapop/model.md#laser.cholera.metapop.model.run_model)`.
- [ ] **Griffe missing-annotation warnings.** Three params in
  `src/laser/cholera/calc_model_likelihood.py` (`date_start`,
  `date_stop`, `epidemic_peaks` at approximately lines 251-254 and
  321-324) and one return annotation in `src/laser/cholera/likelihood.py:603`
  need type annotations. Trivial edits (e.g.,
  `date_start: str | None = None`). Pair with a
  `tests/test_calc_model_likelihood.py` run to confirm no regressions.
- [ ] **Legacy module docstring** in
  `src/laser/cholera/calc_model_likelihood.py`. Leftover narrative from
  the original R port: "Spring likelihood functions for scoring cholera
  model fits…" and "Translation complete. Here's a summary of the key
  design decisions:". mkdocstrings dutifully renders these on the module
  page. Edit out the obsolete framing; keep the technical "Key design
  decisions" content.
- [ ] **Flip the workflow to `--strict`.** In
  `.github/workflows/docs.yml`, change `uvx properdocs build -f mkdocs.yml` to
  `uvx properdocs build -f mkdocs.yml --strict`. Mirror in `tox.ini`'s `[testenv:docs]`.
  After this, §11's "`properdocs build -f mkdocs.yml --strict` green" checkbox closes.

### Optional, defer until needed

- [ ] **PR previews.** Material + a Cloudflare Pages / Netlify /
  preview-deploy GitHub Action can render each PR's docs at a unique
  URL. Worth doing if reviewers actively eyeball doc PRs.
- [ ] **Versioned docs via [`mike`](https://github.com/jimporter/mike).**
  RTD did this for free; on GitHub Pages it needs the plugin. Only
  matters once releases start linking back to versioned docs.
- [ ] **Hand-written reference pages instead of `_gen_reference.py`.**
  The current script *is* the documented mkdocstrings recipe for
  automatic enumeration. Hand-writing ~20 stubs would eliminate the
  custom script at the cost of manual maintenance for every new module.
  Lower-priority; only worth doing if the trade-off feels wrong in
  practice.
