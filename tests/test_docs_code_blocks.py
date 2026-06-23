"""Regression harness for runnable code examples in ``docs/**/*.md``.

For each doc page whose extraction policy in
``tests/docs/extract_doc_blocks.py`` is not ``skip-all``, this test
extracts the page's Python blocks into a single concatenated script under
``tests/docs/scripts/`` and runs it in a sub-process.

Coverage today: 10 pages × 1 script each — index / installation /
changelog / contributing / authors / autogen reference pages contribute
no runnable blocks; explanation pages are illustrative-only;
``how-to/interoperate-with-mosaic.md`` is a placeholder-path stub; and
the two tutorials with structural blockers (`tutorials/single-location.md`
needs the ``mods``-ndarray-coercion follow-up fix, and
`tutorials/multi-location-country.md` needs the unversioned
``tmp/laser-init/`` extract) are skipped via the extractor's policy map.
The ``tutorials/single-location.md`` Step-1 doctest is still verified
independently by the ``doctest-docs`` tox env via
``pytest --doctest-glob='docs/**/*.md'``.

This is a *runtime* check — it confirms each block does not raise. It
does NOT pin output values; the canonical R-vs-Python parity tests
(``tests/R/test-python-parity.R``) and the per-component Python unit
tests cover numerical correctness.

Failure of any test here implies a documentation regression: either a
code block was edited to call a now-removed API, or the underlying
package signature drifted from what the docs claim. Fix by updating
the doc, by tagging the offending block with ``<!-- doc-test:skip -->``
above the fence (single-block opt-out), or by adding the page to
``EXTRACTION_POLICY`` in ``tests/docs/extract_doc_blocks.py``
(``skip-all`` / ``extract-last``).
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "tests" / "docs" / "scripts"
EXTRACTOR = REPO_ROOT / "tests" / "docs" / "extract_doc_blocks.py"


def _scripts() -> list[Path]:
    """Run the extractor once and return the resulting scripts.

    Doing this from inside the test module (rather than at import time) makes
    the test self-contained: edits to a doc page are picked up on the next
    ``pytest`` invocation without a manual extractor run.
    """
    subprocess.run(
        [sys.executable, str(EXTRACTOR), "--out", str(SCRIPTS_DIR)],
        check=True,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
    )
    return sorted(SCRIPTS_DIR.glob("*.py"))


def pytest_generate_tests(metafunc):
    if "script" in metafunc.fixturenames:
        scripts = _scripts()
        metafunc.parametrize("script", scripts, ids=[s.stem for s in scripts])


def test_doc_code_block_runs(script: Path) -> None:
    """Run an extracted documentation script and assert it exits cleanly.

    Given a Python script auto-extracted from a documentation page by
    ``tests/docs/extract_doc_blocks.py``, when the script is executed in a
    fresh subprocess, then the exit code is ``0`` (no uncaught exception).

    Failure implies a documented code example no longer runs against the
    current package. Inspect the captured stderr for the underlying
    traceback; the typical fix is either to update the doc to match the
    current API or to mark the block illustrative via the extractor's
    per-file policy / per-block ``<!-- doc-test:skip -->`` tag.
    """
    # Force the non-interactive matplotlib backend (defence in depth — the
    # auto-generated script preamble already calls
    # `os.environ.setdefault("MPLBACKEND", "Agg")`, but pinning it on the
    # subprocess env catches the case where someone tweaks the extractor
    # header in a way that drops the preamble).
    env = {**os.environ, "MPLBACKEND": "Agg"}
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        pytest.fail(
            f"{script.name} exited with code {proc.returncode}.\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}",
            pytrace=False,
        )
