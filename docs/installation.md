# Installation

LASER Cholera (`laser.cholera`) supports Python 3.10 – 3.14 on Linux, macOS,
and Windows.

## From PyPI

The simplest install pulls the latest release wheel from PyPI:

```bash
pip install laser-cholera
```

After install, the `metapop` command-line entry point is available:

```bash
metapop --help
```

## From source

For development work — running tests, modifying the model, building docs —
clone the repo and install in editable mode. The project uses
[`uv`](https://docs.astral.sh/uv/) for environment management:

```bash
# Install uv itself if you don't already have it.
python3 -m pip install --user uv

# Pick a Python version that matches the project's supported range.
uv python install 3.12

# In the laser-cholera/ working tree:
uv venv
source .venv/bin/activate            # macOS / Linux
# .venv\Scripts\activate              # Windows

uv pip install -e .                  # install runtime deps
uv pip install -e '.[dev]'           # plus test runner + build tools
uv pip install -e '.[docs]'          # plus the MkDocs documentation stack
uv pip install -e '.[nb]'            # plus Jupyter for notebook authoring
```

Multiple extras can be combined: `uv pip install -e '.[dev,docs]'`.

## Verifying the install

The fastest smoke test runs the model with the default parameters bundled
in the wheel:

```bash
metapop --seed 20240101 --loglevel WARNING
```

This runs the full default window (~1155 ticks across 40 patches) and
completes in under ten seconds on a modern laptop. If it raises, the
install is broken.

Shortening the window from the CLI is not currently supported: the
time-series matrices (`b_jt`, `d_jt`, `nu_1_jt`, `nu_2_jt`, `psi_jt`) are
sized from the bundled JSON and would need to be re-sliced to match a
narrower `[date_start, date_stop]`. For a short programmatic run, see the
`sim_duration` + matrix-trim pattern in `tests/test_model.py`.

The full test suite (`pytest tests/`) takes ~10 seconds and covers each
compartment component individually.

## Optional dependencies

- `[docs]` — ProperDocs (MkDocs) + Material + mkdocstrings for building this site locally.
- `[dev]` — pytest, `build`, and `uv` for the development workflow.
- `[nb]` — Jupyter + nbconvert/nbformat for the notebook helpers under the
  scenario authoring path.

See [the contributing guide](contributing.md) for the development workflow.
