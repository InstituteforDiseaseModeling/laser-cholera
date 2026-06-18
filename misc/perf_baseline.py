"""Deterministic baseline / verification harness for the metapop pipeline.

Runs the model with `SEED = 20260618` and the default 13-component
pipeline, hashes every numeric array on `model.people` and
`model.patches` (29 channels), and writes the hashes to
`misc/perf_baseline.json`. After any change that *should* be
numerically invariant — a refactor, a perf optimization, an internal
data-structure swap — re-running with `--verify` proves the change is
bit-for-bit identical to the baseline.

## Staleness contract

The committed `misc/perf_baseline.json` is keyed to a specific
`(seed, default parameter set, component list, laser-core RNG
version, NumPy/BLAS build)` tuple. If you intentionally change any of
those — adopt a new laser-core release that touches the PRNG, swap the
default parameter file, alter the default component pipeline, upgrade
NumPy / switch BLAS, or edit any of the binomial / poisson draw sites in
the compartments — the existing baseline will go stale and `--verify`
will report a (correct, expected) mismatch.

When you knowingly change model dynamics: re-capture and commit the
new baseline in the same patch as the dynamics change, so the next
person who runs `--verify` sees a clean result rather than having to
diff against an obsolete reference:

    python3 misc/perf_baseline.py        # re-capture
    git add misc/perf_baseline.json      # commit alongside the dynamics change

## Usage

    python3 misc/perf_baseline.py             # capture (writes JSON)
    python3 misc/perf_baseline.py --verify    # compare against the JSON
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from laser.cholera.metapop import get_parameters
from laser.cholera.metapop.model import Model
from laser.cholera.metapop.model import run_model  # noqa: F401  ensure side-effects

SEED = 20260618
BASELINE_PATH = Path("misc/perf_baseline.json")


def hash_array(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


PEOPLE_PROPS = ("S", "E", "Isym", "Iasym", "R", "V1", "V2")
PATCHES_PROPS = (
    "births",
    "disease_deaths",
    "new_symptomatic",
    "incidence",
    "incidence_env",
    "incidence_human",
    "Lambda",
    "N",
    "non_disease_deaths",
    "Psi",
    "reported_cases",
    "reported_deaths",
    "spatial_hazard",
    "W",
    "dose_one_doses",
    "dose_two_doses",
    "beta_jt_env",
    "beta_jt_human",
    "delta_jt",
    "coupling",
    "pi_ij",
)


def hash_model(model: Model) -> dict[str, str]:
    """Hash every known NumPy property on `model.people` and `model.patches`.

    `LaserFrame` doesn't surface its registered vector / array
    properties via `dir()`, so the enumeration is by an explicit list
    that mirrors `RInterface.__init__`'s coverage.
    """
    out: dict[str, str] = {}
    for prop in PEOPLE_PROPS:
        if hasattr(model.people, prop):
            out[f"people.{prop}"] = hash_array(getattr(model.people, prop))
    for prop in PATCHES_PROPS:
        if hasattr(model.patches, prop):
            out[f"patches.{prop}"] = hash_array(getattr(model.patches, prop))
    out["log_likelihood"] = (
        repr(model.log_likelihood) if hasattr(model, "log_likelihood") else "<missing>"
    )
    return out


def run() -> Model:
    params = get_parameters()
    params.seed = SEED
    params.quiet = True
    model = Model(params, name="perf-baseline")
    # Import here to break the otherwise-circular components / model import.
    from laser.cholera.metapop import (  # noqa: PLC0415
        Analyzer,
        Census,
        DerivedValues,
        Environmental,
        EnvToHuman,
        Exposed,
        HumanToHuman,
        Infectious,
        Parameters,
        Recorder,
        Recovered,
        Susceptible,
        Vaccinated,
    )
    model.components = [
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
    model.run()
    return model


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Compare against baseline.")
    args = parser.parse_args(argv)

    print(f"Running model with seed={SEED} ...", file=sys.stderr)
    model = run()
    hashes = hash_model(model)

    if args.verify:
        if not BASELINE_PATH.exists():
            print(f"Baseline {BASELINE_PATH} missing; run without --verify first.", file=sys.stderr)
            return 2
        baseline = json.loads(BASELINE_PATH.read_text())
        keys = sorted(set(hashes) | set(baseline))
        mismatches = []
        for k in keys:
            b = baseline.get(k, "<absent>")
            n = hashes.get(k, "<absent>")
            if b != n:
                mismatches.append((k, b, n))
        if not mismatches:
            print(f"OK: {len(hashes)} channels match baseline.")
            return 0
        print(f"MISMATCH: {len(mismatches)} channels differ from baseline:")
        for k, b, n in mismatches:
            print(f"  {k}")
            print(f"    baseline: {b}")
            print(f"    current : {n}")
        return 1

    BASELINE_PATH.write_text(json.dumps(hashes, indent=2, sort_keys=True))
    print(f"Wrote {len(hashes)} channel hashes to {BASELINE_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
