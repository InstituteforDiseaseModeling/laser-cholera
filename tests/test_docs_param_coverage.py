"""Documentation coverage test for the parameter-reference group pages.

This module enforces that **every** top-level field of
`src/laser/cholera/metapop/data/default_parameters.json` is documented in
exactly one of the parameter group pages under
`docs/reference/parameters/`, and that no extra (typo) parameter is
documented on a group page that does not also exist in the JSON. The
test is the source of truth for completeness of the parameter reference
section; the docs themselves are what the test exercises.

The test additionally cross-checks the `scalars` and `arrays` lists in
`src/laser/cholera/metapop/params.py:dict_to_propertysetex` to ensure
that every name those lists reference is also documented — catching the
case where `params.py` learns about a new parameter but the docs do
not.

Failure of any of these tests means the parameter reference is out of
sync with the code: a parameter has been added without a docs entry
(or removed without removing the entry), or a docs entry has drifted
from the JSON. Fix the docs (not the test) — the test is the source of
truth for documentation completeness.
"""

import json
import logging
import re
from pathlib import Path

import pytest

logger = logging.getLogger("laser.cholera")

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS_JSON = REPO_ROOT / "src" / "laser" / "cholera" / "metapop" / "data" / "default_parameters.json"
PARAMS_PY = REPO_ROOT / "src" / "laser" / "cholera" / "metapop" / "params.py"
GROUP_PAGES_DIR = REPO_ROOT / "docs" / "reference" / "parameters"

# Header lines look like `### \`<name>\`` in the group pages. The
# leading `###` is the H3 heading level used uniformly for parameter
# entries (see e.g. run-identity.md).
PARAM_HEADER_RE = re.compile(r"^###\s+`([A-Za-z_][A-Za-z_0-9]*)`\s*$", re.MULTILINE)

# Names listed inside the `scalars = [ ... ]` and `arrays = [ ... ]`
# blocks of `dict_to_propertysetex`. Each entry is a tuple
# `("name", ...)` so we just look for the first string in each tuple.
SCALARS_BLOCK_RE = re.compile(r"scalars\s*=\s*\[(.*?)\]", re.DOTALL)
ARRAYS_BLOCK_RE = re.compile(r"arrays\s*=\s*\[(.*?)\]", re.DOTALL)
TUPLE_NAME_RE = re.compile(r'^\s*\(\s*"([A-Za-z_][A-Za-z_0-9]*)"', re.MULTILINE)


def _load_default_params_keys() -> set[str]:
    """Return the set of top-level keys in `default_parameters.json`.

    Returns:
        Set of every top-level field name in the bundled JSON.
    """
    logger.info("Loading default_parameters.json keys from %s", DEFAULT_PARAMS_JSON)
    text = DEFAULT_PARAMS_JSON.read_text()
    return set(json.loads(text).keys())


def _load_params_py_names() -> set[str]:
    """Extract the parameter names referenced in `params.py`'s scalars and arrays lists.

    Looks for the `scalars = [ ... ]` and `arrays = [ ... ]` blocks
    inside `dict_to_propertysetex` and pulls the leading `"name"` from
    each tuple. Commented-out lines (those starting with `#`) are
    skipped by the regex because they do not begin with `(`.

    Returns:
        Set of parameter names that `params.py` actively coerces.
    """
    logger.info("Extracting scalars/arrays names from %s", PARAMS_PY)
    source = PARAMS_PY.read_text()
    names: set[str] = set()
    for block_re in (SCALARS_BLOCK_RE, ARRAYS_BLOCK_RE):
        match = block_re.search(source)
        if match is None:  # pragma: no cover - defensive
            raise AssertionError(f"Could not locate block matching {block_re.pattern!r} in {PARAMS_PY}")
        block = match.group(1)
        for tuple_match in TUPLE_NAME_RE.finditer(block):
            names.add(tuple_match.group(1))
    return names


def _load_documented_params() -> dict[str, list[str]]:
    """Walk every group page (except `index.md`) and collect parameter headers.

    Returns:
        Mapping from parameter name to the list of group-page filenames
        that document it. A correctly-covered parameter has a
        single-element list; offenders have zero (missing) or two-plus
        (duplicated) entries.
    """
    logger.info("Walking parameter group pages under %s", GROUP_PAGES_DIR)
    documented: dict[str, list[str]] = {}
    for page in sorted(GROUP_PAGES_DIR.glob("*.md")):
        if page.name == "index.md":
            continue
        text = page.read_text()
        for match in PARAM_HEADER_RE.finditer(text):
            name = match.group(1)
            documented.setdefault(name, []).append(page.name)
    return documented


def test_every_default_param_documented_exactly_once() -> None:
    """Each JSON key must appear under one and only one group page.

    Given the bundled `default_parameters.json` and the ten group pages
    under `docs/reference/parameters/`, when we collect every `###
    \\`name\\`` H3 across the group pages (excluding `index.md`), then
    every top-level JSON key must appear in exactly one group page.

    Implication of failure: a parameter is either undocumented (the
    reference section is incomplete and readers cannot look it up) or
    documented in multiple group pages (the reference contradicts
    itself about which feature the parameter belongs to). Fix the docs
    — the test is the source of truth.
    """
    default_keys = _load_default_params_keys()
    documented = _load_documented_params()

    missing = sorted(name for name in default_keys if name not in documented)
    duplicated = sorted((name, pages) for name, pages in documented.items() if len(pages) >= 2 and name in default_keys)

    parts: list[str] = []
    if missing:
        parts.append(f"Parameters in default_parameters.json that are NOT documented in any group page: {missing}")
    if duplicated:
        parts.append(f"Parameters documented in MORE than one group page: {duplicated}")
    assert not parts, "\n".join(parts)


def test_no_documented_param_is_missing_from_defaults() -> None:
    """Every parameter the docs claim to cover must actually exist in the JSON.

    Given the same group-page sweep, when we collect the union of
    documented parameter names, then every one of them must appear as a
    top-level key in `default_parameters.json`.

    Implication of failure: a docs entry refers to a parameter that no
    longer exists (or never existed) in the bundled defaults — likely a
    typo or a stale entry left behind after a rename / removal. Fix the
    docs.
    """
    default_keys = _load_default_params_keys()
    documented = _load_documented_params()

    extras = sorted(name for name in documented if name not in default_keys)
    assert not extras, (
        f"Parameters documented on group pages but absent from default_parameters.json: {extras}. "
        "These are either typos in the docs or stale entries — fix the docs."
    )


def test_params_py_scalars_and_arrays_are_documented() -> None:
    """Every name in `params.py`'s scalars / arrays lists must be documented.

    Given the `scalars` and `arrays` blocks inside
    `dict_to_propertysetex`, when we extract their parameter names,
    then each name must appear under exactly one group page (the same
    invariant as for the JSON keys, applied to the second source of
    truth).

    The `params.py` lists drive the actual dtype coercion; if a name
    here is undocumented, callers will not know what they can set, and
    if it is duplicated they will not know which feature it gates.

    Note: `epidemic_threshold` is intentionally commented out of both
    blocks in `params.py` because it is dispatched dynamically (scalar
    vs array) downstream of the lists. The regex therefore does not
    pick it up, and the test does not need a special case.

    Implication of failure: a feature parameter that `params.py` knows
    how to coerce is missing from the reference, or a name has been
    renamed in one place but not the other. Fix the docs.
    """
    params_names = _load_params_py_names()
    documented = _load_documented_params()

    missing = sorted(name for name in params_names if name not in documented)
    assert not missing, f"Parameters listed in params.py's scalars/arrays blocks but undocumented in any group page: {missing}. Fix the docs."


@pytest.mark.parametrize(
    "expected_group",
    [
        "disease-progression.md",
        "environmental-transmission.md",
        "geography-and-mobility.md",
        "human-transmission.md",
        "initial-populations.md",
        "regime-switching.md",
        "reporting.md",
        "run-identity.md",
        "vaccination.md",
        "vital-dynamics.md",
    ],
)
def test_group_page_exists(expected_group: str) -> None:
    """Each of the ten planned group pages must exist as a markdown file.

    Given the ten parameter groups defined in `misc/standalone-docs-plan.md`,
    when we look under `docs/reference/parameters/`, then each group
    page must exist as a `.md` file.

    Implication of failure: the parameter taxonomy promised by the plan
    has lost a group page (deleted or renamed without updating the
    plan); the index page links into it will 404 in the built site.
    """
    target = GROUP_PAGES_DIR / expected_group
    assert target.exists(), f"Missing group page: {target}"
