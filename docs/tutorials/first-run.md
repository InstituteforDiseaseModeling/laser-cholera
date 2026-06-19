# Your first run

In this tutorial you will run the bundled `laser-cholera` defaults two ways — first from the command line, then from a Python REPL — and then peek at the populated model object to see what a finished simulation actually contains. Plan on about ten minutes. By the end you will have a `Model` instance in memory whose compartment time-series and per-patch reporting outputs you can inspect, plot, or hand off to downstream analysis.

## Prerequisites

You need a working `laser-cholera` install with the `metapop` console script on your `PATH`. If you have not done that yet, follow [Installation](../installation.md) first; the rest of this page assumes:

```bash
metapop --help
```

prints the click usage banner without errors.

## Step 1 — Run the bundled defaults from the CLI

The fastest way to confirm the model runs end-to-end is the `metapop` entry point with no parameter file:

```bash
metapop --seed 20240930 --loglevel INFO -q
```

Three things to notice in that command:

- `--seed 20240930` pins the PRNG. `20240930` is the canonical doc-example seed used throughout these tutorials — it is the date of the first commit to `laser-cholera` and is also the CLI default, so passing it explicitly here is for emphasis, not because you have to.
- `--loglevel INFO` raises the package logger above its default `WARNING` so you can watch the model report what it is doing.
- `-q` (short for `--quiet`) suppresses the per-tick `tqdm` progress bar. The model still runs all ~1155 ticks; you just do not get the rolling bar.

What you should see on stdout: a handful of `INFO` log lines from `laser.cholera` (parameter load, model construction, calendar window, "Running the Cholera Metapop model for 1155 ticks…"), then several seconds of silence while the simulation runs across the bundled ~40 patches, then a per-phase timing summary and an `INFO` "Completed…" line. Total wall time is well under ten seconds on a modern laptop.

If that ran without raising, your install is sound. Now do the same thing from Python so you can keep the result around.

## Step 2 — Repeat in Python

The CLI is a thin wrapper around `run_model`. Run the same simulation programmatically:

```python
from laser.cholera.metapop.model import run_model

model = run_model(None)
```

The single argument is the parameter source. Passing `None` tells `run_model` to load the bundled `default_parameters.json` — exactly what the CLI does when you do not pass `--params`. `run_model` returns the populated `Model` instance after the run completes, so `model` is now a live object whose state buffers are filled in for every tick. The rest of this tutorial inspects that object.

!!! tip
    The same call from a Jupyter notebook will also print a `tqdm` progress bar. Pass `quiet=True` to `run_model` if you want the equivalent of the CLI's `-q`.

## Step 3 — Look at the compartments

The model stores per-compartment, per-patch time-series on `model.people`. The seven compartments are `S`, `E`, `Isym`, `Iasym`, `R`, `V1`, and `V2` — susceptible, exposed, symptomatic infectious, asymptomatic infectious, recovered, and the two vaccinated classes. Each is a NumPy array of shape `(nticks + 1, npatches)`: one extra row at the front for the `t = 0` seed state, then one row per simulation tick.

Start with susceptibles:

```python
S = model.people.S
print(S.shape)
print(S[0])    # initial S per patch (seeded from params.S_j_initial)
print(S[-1])   # S at the final tick
```

The first row is the configured initial condition; the last row is what the simulation produced after running the full window. The same shape and indexing convention apply to every compartment, so you can compare arrivals into `R` against departures from `S` patch-by-patch.

Per-patch derived quantities — the things you would report, not the raw state — live on `model.patches`. Two of the most useful:

```python
incidence = model.patches.incidence
reported_cases = model.patches.reported_cases
print(incidence.shape, reported_cases.shape)
print(reported_cases[-1])  # final-tick reported cases per patch
```

`incidence` is the true per-tick, per-patch new-symptomatic count produced by the transmission components. `reported_cases` applies the reporting fraction `rho` and the reporting lag `delta_reporting_cases` to that, and is the series you would compare against observational data.

!!! tip
    `model.results` is the same data reshaped into the `[location, time]` layout the R reference implementation expects, with the `t = 0` row stripped. Use `model.people` / `model.patches` when you want the raw simulation buffers; use `model.results` when you are cross-checking against MOSAIC's R outputs.

## Step 4 — Make a small chart

Total infectious load over time, summed across patches, is a single line of matplotlib:

```python
import matplotlib.pyplot as plt

infectious_total = model.people.Isym.sum(axis=1) + model.people.Iasym.sum(axis=1)
plt.plot(infectious_total)
plt.xlabel("Tick (day)")
plt.ylabel("Total infectious (Isym + Iasym)")
plt.show()
```

`.sum(axis=1)` collapses the patch dimension, leaving a length-`(nticks + 1)` vector — one value per simulated day. You should see a curve that starts at the initial seeded infections, rises through one or more outbreak peaks, and settles toward an endemic level shaped by birth/death turnover and recovery.

## Where to next

You now have the shape of an end-to-end run. From here:

- To build a parameter set from scratch one knob at a time, work through [Tutorial: single-location](single-location.md). For the underlying configuration to mutate, see [Configurations › Single-location toy](../configurations/single-location.md).
- To override individual parameters without leaving the bundled defaults, see [How-to: override parameters](../how-to/override-parameters.md).
- For the full catalog of every parameter the model accepts, see [Reference › Parameters](../reference/parameters/index.md).
- To understand what the model is actually doing between ticks, see [Explanation › Model overview](../explanation/model-overview.md). For a country-scale configuration to read alongside it, see [Configurations › Multi-admin](../configurations/multi-admin.md).
