## General
- Always note accepted edits in CHANGELOG.md
- Always use double quotes unless already inside a quoted string
- Always use pathlib rather than os.path as long as possible
- Always import logger from .logging and frequently log as INFO level internal actions.
## Documentation - docstrings
- Always use Google style docstrings formatted for markdown (not restructuredtext)
- Always include information on exceptions explicitly raised in the code
- Consider executable code examples in docstrings when appropriate and concise
## Testing
- Always write tests in a "given-when-then" style: given this scenario, when I call this function, then I expect this result
- Always add a docstring to tests explaining the purpose of the test and the implications of failure(s)
- Always comment on inconsistencies or ambiguities in functions being tested
- Always run new tests to verify implementation before considering implementation as complete
## Working files
- The repo-root `misc/` directory is where ad-hoc working documents live: red-team assessments, project plans, todo / checklist files, one-off analyses, and "temporary" tools (small scripts that support a specific change but aren't part of the package or the test suite). Prefer it over the repo root for new artifacts of those kinds. Contents *can* be committed (the directory is tracked at the file level — `git add misc/whatever` to include something in a PR), but they sit outside the package's quality gates: `misc/` is excluded from ruff (`[tool.ruff] extend-exclude` in `pyproject.toml`), is not on the MkDocs `docs_dir`, and is not in pytest's `testpaths` — so linting / docs / doctest collection ignore it.
- The gitignored `tmp/` directory is the home for files that should *never* be committed: throwaway scratch, one-shot intermediate outputs, anything you'd be unhappy to see in a PR. If a file outgrows `tmp/` (becomes worth saving across sessions), move it to `misc/`; if it's a finished tool or analysis, decide whether to `git add` it.

## Linting
- Always run `uvx --with tox-uv tox -e check` and resolve all failures before committing or proposing changes to be committed. The hook runs the pre-commit suite (ruff lint, ruff format, trailing whitespace, end-of-file fixer, debug-statement check); both the work and any follow-up edits the hook auto-applies must be clean.
- Prefer `pytest.raises(ExceptionType, match=r"...")` over `self.assertRaises(...)`; ruff's PT011 rule flags bare `pytest.raises(ValueError)` without a `match` pattern.
