"""Smoke tests for the three documented starter configurations.

Standalone-docs plan section 9 calls for a smoke test that loads each of the
three shipped configurations, runs a short simulation, and confirms the model
returns without raising. The configurations are the user-facing entry points
into ``laser-cholera`` — single-location toy, single-country multi-admin
extract, and the bundled SSA baseline — so the cost of a silent regression is
that the documentation example a new user copy-pastes blows up in their face.

Each test follows the given-when-then style: given a configuration on disk and
a short simulation window, when the parameters are loaded and a Model is built
and (where cost permits) run, then the run completes and the canonical
``model.people.S`` array exists with the expected ``(nticks + 1, npatches)``
shape. Tests do NOT depend on ``tmp/laser-init/`` — only on files committed to
the repo.
"""

from datetime import timedelta
from pathlib import Path

import numpy as np

from laser.cholera.metapop import Analyzer
from laser.cholera.metapop import Census
from laser.cholera.metapop import DerivedValues
from laser.cholera.metapop import Environmental
from laser.cholera.metapop import EnvToHuman
from laser.cholera.metapop import Exposed
from laser.cholera.metapop import HumanToHuman
from laser.cholera.metapop import Infectious
from laser.cholera.metapop import Parameters
from laser.cholera.metapop import Recorder
from laser.cholera.metapop import Recovered
from laser.cholera.metapop import Susceptible
from laser.cholera.metapop import Vaccinated
from laser.cholera.metapop import get_parameters
from laser.cholera.metapop.model import Model

REPO_ROOT = Path(__file__).resolve().parents[1]
SINGLE_LOCATION_JSON = REPO_ROOT / "docs" / "configurations" / "code" / "single-location.json"
MULTI_ADMIN_JSON_GZ = REPO_ROOT / "docs" / "configurations" / "code" / "multi-admin.json.gz"
SSA_BASELINE_JSON = REPO_ROOT / "src" / "laser" / "cholera" / "metapop" / "data" / "default_parameters.json"

COMPONENTS = [
    Susceptible,
    Exposed,
    Recovered,
    Infectious,
    Vaccinated,
    Census,
    HumanToHuman,
    EnvToHuman,
    Environmental,
    DerivedValues,
    Analyzer,
    Recorder,
    Parameters,
]


def _shorten_to_nticks(params, nticks: int) -> None:
    """Mutate ``params`` in place to a ``nticks``-tick simulation window.

    The validator (already passed during the ``get_parameters`` call) enforces
    that every ``(nticks, npatches)`` matrix matches the calendar window, so
    when we shrink the window we have to slice the matrices to match. Five
    time-major matrices live in the parameter set: ``b_jt``, ``d_jt``,
    ``nu_1_jt``, ``nu_2_jt``, ``psi_jt``. The pattern mirrors
    ``tests/test_model.py::TestModel.get_test_parameters``.
    """
    params.nticks = nticks
    params.date_stop = params.date_start + timedelta(days=nticks - 1)
    params.b_jt = params.b_jt[:nticks, :]
    params.d_jt = params.d_jt[:nticks, :]
    params.nu_1_jt = params.nu_1_jt[:nticks, :]
    params.nu_2_jt = params.nu_2_jt[:nticks, :]
    params.psi_jt = params.psi_jt[:nticks, :]


def _build_model(params, name: str) -> Model:
    """Construct a Model with the canonical 13-component pipeline.

    The pipeline mirrors ``misc/perf_baseline.py``; using the same ordering
    means the smoke test fails the same way the baseline does when a component
    contract changes.
    """
    model = Model(params, name=name)
    model.components = COMPONENTS
    return model


def test_single_location_runs():
    """Single-location toy configuration loads, builds, and runs without raising.

    Given the committed ``docs/configurations/code/single-location.json``
    (one patch, mobility / environmental transmission / vaccination /
    seasonality off),
    when ``get_parameters`` loads it, the simulation window is shortened to
    30 ticks, a Model is constructed with the canonical pipeline, and
    ``model.run()`` is invoked,
    then the run completes without raising and ``model.people.S`` exists with
    shape ``(31, 1)`` — i.e. ``nticks + 1`` rows by one patch.

    Failure implies the published single-location config has drifted from the
    parameter / validator / component contract and the first thing a new user
    pastes from the docs is broken.
    """
    params = get_parameters(SINGLE_LOCATION_JSON)
    _shorten_to_nticks(params, 30)

    model = _build_model(params, name="docs-single-location")
    model.run()

    assert hasattr(model.people, "S"), "Susceptible component did not allocate model.people.S"
    assert model.people.S.shape == (31, 1), f"Unexpected S shape {model.people.S.shape}"


def test_multi_admin_runs():
    """Multi-admin (Mozambique adm2) configuration loads, builds, and runs without raising.

    Given the committed ``docs/configurations/code/multi-admin.json.gz``
    (157 adm2 districts, mobility + seasonality + both transmission routes on),
    when ``get_parameters`` loads it, the simulation window is shortened to
    30 ticks, a Model is constructed with the canonical pipeline, and
    ``model.run()`` is invoked,
    then the run completes without raising and ``model.people.S`` exists with
    shape ``(31, 157)`` — ``nticks + 1`` rows by ``npatches`` columns.

    Failure implies the frozen ``laser-init`` extract or the parameter
    contract has drifted; the multi-admin documentation page would no longer
    reproduce.
    """
    params = get_parameters(MULTI_ADMIN_JSON_GZ)
    npatches = len(params.location_name)
    assert npatches == 157, f"Expected 157 Mozambique adm2 districts, got {npatches}"

    _shorten_to_nticks(params, 30)

    model = _build_model(params, name="docs-multi-admin")
    model.run()

    assert hasattr(model.people, "S"), "Susceptible component did not allocate model.people.S"
    assert model.people.S.shape == (31, npatches), f"Unexpected S shape {model.people.S.shape}"
    assert np.all(model.people.S >= 0), "Susceptible counts must be non-negative"


def test_ssa_baseline_loads():
    """SSA baseline configuration loads and constructs a Model without raising.

    Given the bundled ``src/laser/cholera/metapop/data/default_parameters.json``
    (the Sub-Saharan-Africa country-level baseline that ships with the
    package),
    when ``get_parameters`` loads it, the simulation window is shortened to
    5 ticks, and a Model is constructed with the canonical pipeline,
    then construction completes without raising and ``model.people.S`` is
    allocated with shape ``(6, npatches)``. We deliberately do NOT call
    ``model.run()`` here — the SSA baseline has many patches and even a
    5-tick run is the most expensive of the three configurations; the
    standalone-docs plan permits skipping the run when running 5 ticks is
    too slow.

    Failure implies the bundled defaults no longer round-trip through the
    parameter loader + component pipeline, which would break virtually every
    docs example.
    """
    params = get_parameters(SSA_BASELINE_JSON)
    npatches = len(params.location_name)

    _shorten_to_nticks(params, 5)

    model = _build_model(params, name="docs-ssa-baseline")

    assert hasattr(model.people, "S"), "Susceptible component did not allocate model.people.S"
    assert model.people.S.shape == (6, npatches), f"Unexpected S shape {model.people.S.shape}"
