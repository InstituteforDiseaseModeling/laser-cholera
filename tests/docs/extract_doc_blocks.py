"""Walk docs/**/*.md, extract Python code blocks, and emit a per-page script.

Two block kinds are extracted:

- **Doctest blocks** — fenced ``python`` or ``pycon`` blocks whose first
  non-blank line starts with ``>>> ``. These are already executed by the
  ``[testenv:doctest-docs]`` tox env via ``pytest --doctest-glob`` and are
  intentionally skipped by this extractor — there is no value in running them
  twice.

- **Plain-script blocks** — fenced ``python`` blocks without ``>>>`` prompts.
  These are illustrative-but-runnable Python (parameter overrides, model
  builds, tutorial walk-throughs). They get concatenated, in document order,
  into one ``tests/docs/scripts/<slug>.py`` per source file.

A few blocks are demonstrative-only (they show partial state, intentionally
broken code, or shell sessions like ``$ metapop ...``) and are tagged via a
preceding ``<!-- doc-test:skip -->`` comment in the markdown. The extractor
honours that tag and emits the block as a Python comment in the output script
so the doctest harness still sees it for context but doesn't try to run it.

Bash / R / shell blocks are also skipped.

Usage:

    python3 tests/docs/extract_doc_blocks.py [--out tests/docs/scripts]

The extractor is invoked as part of ``tests/test_docs_code_blocks.py``; you
do not normally need to run it by hand.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
DEFAULT_OUT = REPO_ROOT / "tests" / "docs" / "scripts"

# Skip the autogen reference subtree; it's generated from docstrings, not
# hand-authored doc prose with code examples.
SKIP_RELATIVE_DIRS = (
    "reference/metapop",
    "reference/calc_log_likelihood_distributions.md",
    "reference/calc_model_likelihood.md",
    "reference/core.md",
    "reference/index.md",
    "reference/iso_codes.md",
    "reference/utils.md",
)

# Per-file extraction policy. Default is `extract-all` — every Python block on
# the page is concatenated, in order, into one runnable script.
#
# - `skip-all` — page only contains illustrative-but-not-runnable snippets
#   (e.g. internal code excerpts in Explanation pages, placeholder paths).
# - `extract-last` — only the final Python block on the page is runnable
#   (the canonical "Full example" / "Running it" snippet at the bottom of
#   each How-to). The numbered-step blocks above it are illustrative
#   recipe fragments that build a shared `mods` / `params_dict` symbol
#   table the reader carries forward — they are not standalone scripts.
EXTRACTION_POLICY: dict[str, str] = {
    # Explanation pages — narrative + math; code excerpts are illustrative.
    "explanation/mobility.md": "skip-all",
    "explanation/model-overview.md": "skip-all",
    "explanation/seasonality.md": "skip-all",
    "explanation/transmission.md": "skip-all",
    # One-page stub — uses a placeholder path that doesn't resolve.
    "how-to/interoperate-with-mosaic.md": "skip-all",
    # How-tos — only the Full example block at the bottom is meant to be a
    # complete, copy-pasteable script. The earlier Steps blocks build up
    # the recipe incrementally and aren't standalone.
    "how-to/calibrate-and-score.md": "extract-last",
    "how-to/configure-mobility.md": "extract-last",
    "how-to/enable-seasonality.md": "extract-last",
    "how-to/enable-vaccination.md": "extract-last",
    "how-to/override-parameters.md": "extract-last",
    # Tutorials with structural blockers: `single-location.md` walks through a
    # progressive `mods` build whose first 14 blocks set scalar / list values
    # that crash `validate_parameters` until the final block converts them to
    # `np.array(...)` (documented as a follow-up bug in
    # `misc/standalone-docs-plan.md` §14). `multi-location-country.md` depends
    # on an unversioned `tmp/laser-init/mozambique-adm2/` data extract that is
    # gitignored. The Step-1 doctest of the former is still verified by the
    # `doctest-docs` tox env via `pytest --doctest-glob`.
    "tutorials/single-location.md": "skip-all",
    "tutorials/multi-location-country.md": "skip-all",
}

FENCE_RE = re.compile(
    # Capture optional leading indent on the opening fence so we can strip
    # the same prefix from every body line — this is what markdown does when
    # a code block is nested inside a numbered list item (the fence itself
    # is indented to align with the list-item content). Without this dedent
    # the extracted body starts with 4 spaces and Python rejects it as an
    # `IndentationError`.
    r"(?P<indent>[ \t]*)```(?P<lang>[A-Za-z]+)?\s*\n(?P<body>.*?)(?<=\n)(?P=indent)```",
    re.DOTALL,
)

SKIP_TAG = "<!-- doc-test:skip -->"


def _dedent_body(indent: str, body: str) -> str:
    if not indent:
        return body
    stripped: list[str] = []
    for line in body.splitlines(keepends=True):
        if line.startswith(indent):
            stripped.append(line[len(indent) :])
        elif line.strip() == "":
            stripped.append(line.lstrip(" \t"))
        else:
            # Mixed indent — give up dedenting this block to avoid corrupting
            # blocks that intentionally start at column 0 inside a list item.
            return body
    return "".join(stripped)


def _slug(path: Path) -> str:
    rel = path.relative_to(DOCS_DIR)
    return str(rel.with_suffix("")).replace("/", "_")


def _is_doctest_block(body: str) -> bool:
    for line in body.splitlines():
        s = line.strip()
        if not s:
            continue
        return s.startswith(">>>")
    return False


def _strip_doctest_prompts(body: str) -> str:
    """Strip leading ``>>> `` / ``... `` prompts from a doctest block.

    Leaves expected-output lines (lines following a doctest statement that
    don't themselves start with a prompt) in place but commented out — they
    are Python-invalid as bare expressions, so we drop the un-prompted lines
    once we hit them, until the next ``>>>`` prompt resumes. This recovers
    the executable statements while leaving the expected output as comments
    for context.
    """
    out: list[str] = []
    in_statement = False
    for line in body.splitlines():
        stripped = line.lstrip()
        if stripped.startswith(">>> "):
            out.append(stripped[4:])
            in_statement = True
        elif stripped.startswith(">>>"):  # bare `>>>` continuation prompt
            out.append(stripped[3:].lstrip())
            in_statement = True
        elif stripped.startswith("... "):
            out.append(stripped[4:])
        elif stripped.startswith("..."):
            out.append(stripped[3:].lstrip())
        elif stripped == "":
            out.append("")
            in_statement = False
        else:
            # Expected output following a statement — preserve as comment.
            if in_statement:
                out.append(f"# expected: {line.rstrip()}")
            else:
                out.append(line)
    return "\n".join(out)


def _preceding_skip_tag(text: str, start: int) -> bool:
    """Walk back from ``start`` and check whether the nearest non-blank
    non-fence line above is a ``<!-- doc-test:skip -->`` comment."""
    chunk = text[max(0, start - 400) : start]
    lines = chunk.splitlines()
    for line in reversed(lines):
        s = line.strip()
        if not s:
            continue
        if s.startswith("```"):
            continue
        return s == SKIP_TAG
    return False


def extract(md_path: Path) -> tuple[list[str], list[str]]:
    """Return ``(runnable_blocks, skipped_blocks)`` for ``md_path``.

    ``runnable_blocks`` are plain ``python`` fenced blocks without ``>>>``
    prompts and without a preceding ``doc-test:skip`` tag.
    ``skipped_blocks`` are the bodies that were intentionally dropped, so
    callers can report on coverage.
    """
    text = md_path.read_text(encoding="utf-8")
    runnable: list[str] = []
    skipped: list[str] = []
    for m in FENCE_RE.finditer(text):
        lang = (m.group("lang") or "").lower()
        body = _dedent_body(m.group("indent"), m.group("body"))
        # `pycon` is the conventional fence for `>>>` doctest content; treat
        # it as Python for the purpose of skip-reporting.
        if lang not in {"python", "py", "pycon"}:
            continue
        if _preceding_skip_tag(text, m.start()):
            skipped.append(body)
            continue
        # `pycon` blocks (or `python` blocks that happen to use `>>>`) are
        # also executed by the doctest-docs tox env. We still include them
        # here — with prompts stripped — so subsequent plain-Python blocks
        # on the same page can reference the symbols they introduce. The
        # cost of running them twice is small; the alternative is a
        # tutorial whose later blocks can't find `get_parameters` because
        # the import was in an earlier pycon block.
        if _is_doctest_block(body):
            body = _strip_doctest_prompts(body)
        runnable.append(body)
    return runnable, skipped


def emit(out_dir: Path) -> dict[str, dict[str, int]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clean previous runs so removed source blocks don't leave stale scripts.
    for stale in out_dir.glob("*.py"):
        stale.unlink()
    summary: dict[str, dict[str, int]] = {}
    for md_path in sorted(DOCS_DIR.rglob("*.md")):
        rel = md_path.relative_to(DOCS_DIR).as_posix()
        if any(rel.startswith(skip) for skip in SKIP_RELATIVE_DIRS):
            continue
        policy = EXTRACTION_POLICY.get(rel, "extract-all")
        runnable, skipped = extract(md_path)
        if policy == "skip-all":
            skipped.extend(runnable)
            runnable = []
        elif policy == "extract-last" and len(runnable) > 1:
            skipped.extend(runnable[:-1])
            runnable = runnable[-1:]
        summary[rel] = {"runnable": len(runnable), "skipped": len(skipped), "policy": policy}
        if not runnable:
            continue
        slug = _slug(md_path)
        script = out_dir / f"{slug}.py"
        header = [
            '"""Auto-generated runnable concatenation of Python blocks in',
            f"docs/{rel}.",
            "",
            "Regenerated by tests/docs/extract_doc_blocks.py — do not hand-edit.",
            "",
            f"Blocks extracted: {len(runnable)} runnable, {len(skipped)} skipped",
            "(doctest blocks are excluded because the doctest-docs tox env",
            "already runs them via `pytest --doctest-glob`).",
            '"""',
            "",
            "from __future__ import annotations",
            "",
            "# Force a non-interactive matplotlib backend BEFORE any `import",
            "# matplotlib` runs further down. `plt.show()` becomes a no-op and",
            "# no GUI window is opened — needed because some doc blocks render",
            "# trajectories with `matplotlib.pyplot`. `setdefault` so a caller",
            "# who explicitly wants an interactive run can pass MPLBACKEND=...",
            "# in their shell env.",
            "import os",
            "",
            'os.environ.setdefault("MPLBACKEND", "Agg")',
            "",
        ]
        parts = ["\n".join(header)]
        for i, body in enumerate(runnable):
            parts.append(f"\n# ---- block {i + 1} ----\n{body.rstrip()}\n")
        script.write_text("\n".join(parts), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Output dir for extracted scripts.")
    args = parser.parse_args()
    summary = emit(Path(args.out))
    total_runnable = sum(v["runnable"] for v in summary.values())
    total_skipped = sum(v["skipped"] for v in summary.values())
    pages_with_runnable = sum(1 for v in summary.values() if v["runnable"] > 0)
    print(f"Pages scanned: {len(summary)}")
    print(f"Pages with runnable blocks: {pages_with_runnable}")
    print(f"Runnable Python blocks: {total_runnable}")
    print(f"Skipped blocks (doctest + tagged): {total_skipped}")
    for rel, counts in sorted(summary.items()):
        if counts["runnable"] or counts["skipped"]:
            policy = counts.get("policy", "extract-all")
            print(f"  {rel} [{policy}]: runnable={counts['runnable']}, skipped={counts['skipped']}")


if __name__ == "__main__":
    main()
