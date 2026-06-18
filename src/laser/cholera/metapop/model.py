"""Metapopulation model orchestration: `Model`, `RInterface`, and the `cli_run` / `run_model` entry points.

Contains the runtime glue that turns a parameter set plus an ordered
list of component classes into a complete simulation run:

- [`Model`][laser.cholera.metapop.model.Model] — owns the PRNG,
  the `people` / `patches` `LaserFrame` instances, and the per-phase
  timing metrics. Its `components` setter instantiates each component
  against the model, registers `__call__`-bearing ones as run phases,
  and runs each component's `check()` before the first tick.
- [`RInterface`][laser.cholera.metapop.model.RInterface] — a thin
  read-only view object exposing each compartment / patch series
  trimmed of the `t=0` slot and transposed into the `[location, time]`
  layout the R reference implementation expects.
- [`cli_run`][laser.cholera.metapop.model.cli_run] — the
  click-decorated `metapop` console entry point.
- [`run_model`][laser.cholera.metapop.model.run_model] — the canonical
  Python entry point used by tests and by `cli_run`.
"""

import logging
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Optional
from typing import Union

import click
import pandas as pd
from laser.core.laserframe import LaserFrame
from laser.core.propertyset import PropertySet
from laser.core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

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
from laser.cholera.metapop import scenario
from laser.cholera.metapop.utils import UnknownOverrideKey
from laser.cholera.metapop.utils import override_helper

logger = logging.getLogger("laser.cholera")


class RInterface:
    """A simple interface to store results trimmed and transposed for R."""

    def __init__(self, model: "Model"):
        """Snapshot every per-tick / per-patch series as an R-style trimmed-and-transposed view.

        For each available compartment in `model.people` and each
        per-patch property in `model.patches`, attaches a view onto
        `self` named after the property. Time-series (shape
        `(nticks + 1, npatches)`) are sliced `[1:, :]` to drop the
        `t=0` seed row and then transposed to `[npatches, nticks]`.
        Dose-count vectors (`dose_one_doses`, `dose_two_doses`) are
        already `nticks`-shaped, so they are only transposed.
        Per-tick environmental / seasonality matrices (`beta_jt_env`,
        `beta_jt_human`, `delta_jt`) are likewise just transposed.
        The `pi_ij` and `coupling` matrices are passed through
        un-transposed (the latter is symmetric).

        These are *views*, not copies — the underlying buffers are the
        ones the simulation writes into.

        Args:
            model: The completed (or in-progress) `Model` instance.
        """
        # self.S = model.people.S[1:, :].T
        # self.E = model.people.E[1:, :].T
        # self.Isym = model.people.Isym[1:, :].T
        # self.Iasym = model.people.Iasym[1:, :].T
        # self.R = model.people.R[1:, :].T
        # self.V1 = model.people.V1[1:, :].T
        # self.V2 = model.people.V2[1:, :].T

        # Trim the first column (t=0) and transpose for R compatibility
        for compartment in ["S", "E", "Isym", "Iasym", "R", "V1", "V2"]:
            if hasattr(model.people, compartment):
                attr = getattr(model.people, compartment)
                setattr(self, compartment, attr[1:, :].T)

        # self.births = model.patches.births[1:, :].T
        # self.disease_deaths = model.patches.disease_deaths[1:, :].T
        # self.new_symptomatic = model.patches.new_symptomatic[1:, :].T
        # self.incidence = model.patches.incidence[1:, :].T
        # self.incidence_env = model.patches.incidence_env[1:, :].T
        # self.incidence_human = model.patches.incidence_human[1:, :].T
        # self.Lambda = model.patches.Lambda[1:, :].T
        # self.N = model.patches.N[1:, :].T
        # self.non_disease_deaths = model.patches.non_disease_deaths[1:, :].T
        # self.Psi = model.patches.Psi[1:, :].T
        # self.reported_cases = model.patches.reported_cases[1:, :].T
        # self.reported_deaths = model.patches.reported_deaths[1:, :].T
        # self.spatial_hazard = model.patches.spatial_hazard[1:, :].T
        # self.W = model.patches.W[1:, :].T

        # Trim the first column (t=0) and transpose for R compatibility
        for prop in [
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
        ]:
            if hasattr(model.patches, prop):
                attr = getattr(model.patches, prop)
                setattr(self, prop, attr[1:, :].T)

        # self.dose_one_doses = model.patches.dose_one_doses[:].T
        # self.dose_two_doses = model.patches.dose_two_doses[:].T

        # Transpose these for R compatibility
        # Hmm, should these not be nticks+1 in length?
        for prop in ["dose_one_doses", "dose_two_doses"]:
            if hasattr(model.patches, prop):
                attr = getattr(model.patches, prop)
                setattr(self, prop, attr.T)

        # self.beta_jt_env = model.patches.beta_jt_env.T
        # self.beta_jt_human = model.patches.beta_jt_human.T
        # self.delta_jt = model.patches.delta_jt.T

        # Transpose these for R compatibility
        for prop in ["beta_jt_env", "beta_jt_human", "delta_jt"]:
            if hasattr(model.patches, prop):
                attr = getattr(model.patches, prop)
                setattr(self, prop, attr.T)

        # self.coupling = model.patches.coupling

        # Coupling could be in the list above, but coupling is symmetric, so we don't need to transpose it.
        # pi_ij doesn't need transposing.
        for prop in ["coupling", "pi_ij"]:
            if hasattr(model.patches, prop):
                attr = getattr(model.patches, prop)
                setattr(self, prop, attr)

        return


class Model:
    """Top-level metapopulation simulation object: owns state, the PRNG, and the run loop.

    A `Model` is constructed from a `PropertySetEx` of parameters; the
    caller then assigns a list of component classes to `model.components`
    (which instantiates each, registers `__call__`-bearing instances as
    run phases, and validates prerequisites via each component's
    `check()`). Calling `model.run()` advances the simulation
    `nticks` times, building up per-phase timing metrics and the R-style
    [`RInterface`][laser.cholera.metapop.model.RInterface] view on
    `model.results`.

    Attributes:
        params: The full parameter `PropertySetEx`.
        name: Human-readable label used in logs and the PDF filename.
        scenario: The MOSAIC scenario `GeoDataFrame`
            ([`laser.cholera.metapop.scenario.scenario`][]).
        prng: Seeded LASER core PRNG instance shared by every component.
        people: `LaserFrame` of length `npatches` holding the
            compartment time-series (one row per patch, vector-property
            per compartment).
        patches: `LaserFrame` of length `npatches` holding per-patch
            inputs and reporting outputs.
        tinit / tstart / tfinish: Wall-clock timestamps for model
            construction, run start, and run finish respectively
            (`datetime.datetime`, naive — local time).
        metrics: After `run()`, a list of per-tick `(tick, *phase_µs)`
            timing rows.
        results: After `run()`, an
            [`RInterface`][laser.cholera.metapop.model.RInterface] view
            onto the completed simulation arrays.
    """

    def __init__(self, parameters: PropertySet, name: str = "Cholera Metapop"):
        """Construct a fresh model: seed the PRNG and allocate `people` / `patches` frames.

        Does NOT install components — assign `model.components = [...]`
        after construction to wire up the pipeline.

        Args:
            parameters: A validated `PropertySetEx` (typically from
                [`get_parameters`][laser.cholera.metapop.params.get_parameters])
                providing at minimum `seed`, `location_name`, and
                `nticks`.
            name: Display name used in logging and visualization output.
        """
        self.tinit = datetime.now(tz=None)  # noqa: DTZ005
        logger.info(f"{self.tinit}: Creating the {name} model…")
        self.params = parameters
        self.name = name

        self.scenario = scenario

        self.prng = seed_prng(parameters.seed if parameters.seed is not None else self.tinit.microsecond)

        logger.info(f"Initializing the {name} model with {len(parameters.location_name)} patches…")

        # https://gilesjohnr.github.io/MOSAIC-docs/model-description.html

        # setup the LaserFrame for people/population (states and dynamics)
        # setup the LaserFrame for patches (inputs and reporting)
        npatches = len(parameters.location_name)
        self.people = LaserFrame(npatches)
        self.patches = LaserFrame(npatches)

        return

    @property
    def components(self) -> list:
        """
        Retrieve the list of model components.

        Returns:

            list: A list containing the components.
        """

        return self._components

    @components.setter
    def components(self, components: list) -> None:
        """
        Sets up the components of the model and initializes instances and phases.

        This function takes a list of component types, creates an instance of each, and adds each callable component to the phase list.
        It also registers any components with an `on_birth` function with the `Births` component.

        Args:

            components (list): A list of component classes to be initialized and integrated into the model.

        Returns:

            None
        """

        self._components = components
        self.instances = []  # instantiated instances of components
        self.phases = []  # callable phases of the model
        for component in components:
            instance = component(self)
            self.instances.append(instance)
            if "__call__" in dir(instance):
                logger.debug(f"Adding {type(instance).__name__} to the model…")
                self.phases.append(instance)

        _ = [instance.check() for instance in self.instances]

        return

    def run(self) -> None:
        """
        Execute the model for a specified number of ticks, recording the time taken for each phase.

        This method initializes the start time, iterates over the number of ticks specified in the model parameters,
        and for each tick, it executes each phase of the model while recording the time taken for each phase.

        The metrics for each tick are stored in a list. After completing all ticks, it records the finish time and,
        logs a summary of the timing metrics.

        Attributes:

            tstart (datetime): The start time of the model execution.
            tfinish (datetime): The finish time of the model execution.
            metrics (list): A list of timing metrics for each tick and phase.

        Returns:

            None
        """

        self.tstart = datetime.now(tz=None)  # noqa: DTZ005
        logger.info(f"{self.tstart}: Running the {self.name} model for {self.params.nticks} ticks…")

        # The results are just views onto existing NumPy arrays, so we can
        # initialize this here. The Analyzer will need it on the last tick.
        self.results = RInterface(self)

        self.metrics = []

        for tick in tqdm(range(self.params.nticks), desc="Running model", disable=self.params.quiet):
            timing = [tick]
            for phase in self.phases:
                tstart = datetime.now(tz=None)  # noqa: DTZ005
                phase(self, tick)
                tfinish = datetime.now(tz=None)  # noqa: DTZ005
                delta = tfinish - tstart
                timing.append(delta.seconds * 1_000_000 + delta.microseconds)
            self.metrics.append(timing)

        self.tfinish = datetime.now(tz=None)  # noqa: DTZ005
        logger.info(f"{self.tfinish}: Completed the {self.name} model")

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()
        width = max(map(len, sum_columns.index))
        for key in sum_columns.index:
            logger.info(f"{key:{width}}: {sum_columns[key]:13,} µs")
        logger.info("=" * (width + 2 + 13 + 3))
        logger.info(f"{'Total:':{width + 1}} {sum_columns.sum():13,} microseconds")

        return

    def visualize(self, pdf: bool = True) -> Optional[str]:  # pragma: no cover
        """
        Visualize each compoonent instances either by displaying plots or saving them to a PDF file.

        Parameters:

            pdf (bool): If True, save the plots to a PDF file. If False, display the plots interactively. Default is True.

        Returns:

            None
        """

        filename = None

        _debugging = None  # [DerivedValues]

        if not pdf:
            for instance in [self, *self.instances]:
                if (_debugging is None) or (type(instance) in _debugging):
                    if hasattr(instance, "plot"):
                        for _plot in instance.plot():
                            plt.tight_layout()
                            logger.debug(f"Plotting {type(instance).__name__}…")
                            plt.show()
                    else:
                        logger.warning(f"{type(instance).__name__} does not have a plot method.")
                else:
                    logger.debug(f"Skipping {type(instance).__name__} visualization…")

        else:
            logger.info("Generating PDF output…")
            pdf_filename = f"{self.name} {self.tstart:%Y-%m-%d %H%M%S}.pdf"
            with PdfPages(pdf_filename) as pdf:
                for instance in [self, *self.instances]:
                    if (_debugging is None) or (type(instance) in _debugging):
                        if hasattr(instance, "plot"):
                            for title in instance.plot():
                                plt.title(title)
                                plt.tight_layout()
                                logger.debug(f"Plotting {type(instance).__name__}…")
                                pdf.savefig()
                                plt.close()
                        else:
                            logger.warning(f"{type(instance).__name__} does not have a plot method.")
                    else:
                        logger.debug(f"Skipping {type(instance).__name__} visualization…")

            logger.info(f"PDF output saved to '{pdf_filename}'.")
            filename = pdf_filename

        return filename

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield two top-level scenario figures: patch map and per-phase timing pie.

        The first figure overlays a scatter of patch centroids
        (sized / colored by population) on the scenario `GeoDataFrame`
        polygons. The second figure pie-charts the total microseconds
        spent in each run phase across the entire simulation.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            Two labels in order:
            `"Scenario Patches and Populations"` and
            `"Update Phase Times (Total N µsec)"` (where `N` is the
            formatted total).
        """
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Scenario Patches and Populations") if fig is None else fig

        if "geometry" in self.scenario.columns:
            ax = plt.gca()
            self.scenario.plot(ax=ax)
        scatter = plt.scatter(
            self.scenario.longitude,
            self.scenario.latitude,
            s=self.scenario.population / 100_000,
            c=self.scenario.population,
            cmap="inferno",
        )
        plt.colorbar(scatter, label="Population")

        yield "Scenario Patches and Populations"

        metrics = pd.DataFrame(self.metrics, columns=["tick"] + [type(phase).__name__ for phase in self.phases])
        plot_columns = metrics.columns[1:]
        sum_columns = metrics[plot_columns].sum()

        _fig = plt.figure(figsize=(12, 9), dpi=128, num=f"Update Phase Times (Total {sum_columns.sum():,} µsec)") if fig is None else fig

        plt.pie(
            sum_columns,
            labels=sum_columns.index,  # [name for name in sum_columns.index],
            autopct="%1.1f%%",
            startangle=140,
        )

        yield f"Update Phase Times (Total {sum_columns.sum():,} µsec)"
        return


@click.command()
@click.option("--seed", type=int, default=20241107, help="Random seed")
@click.option("--viz", "visualize", is_flag=True, default=False, help="Display visualizations")
@click.option("--pdf", is_flag=True, default=False, help="Output visualization results as a PDF")
@click.option(
    "--outdir",
    "-o",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path.cwd(),
    help="Output directory for results",
)
@click.option(
    "--params",
    "-p",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="JSON (or .json.gz) file with parameters",
)
@click.option("--over", multiple=True, help="Additional parameter overrides (param:value or param=value)")
@click.option(
    "--loglevel",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False),
    default="WARNING",
    help="Logging level",
)
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress console progress output")
@click.option("--hdf5-output", "hdf5_output", is_flag=True, default=False, help="Write per-tick outputs to an HDF5 file via the Recorder")
@click.option("--compress", is_flag=True, default=False, help="Gzip the HDF5 output (only meaningful with --hdf5-output)")
def cli_run(params: Optional[Path], **kwargs: object) -> None:
    """Run the cholera model simulation with the given parameters.

    Initializes the model, sets up the default component pipeline, seeds
    initial infections, runs the simulation, and optionally renders
    visualizations.
    \f

    Args:
        params: Path to a parameters JSON file (or `.json.gz`). When
            `None`, the bundled `default_parameters.json` is used.
        **kwargs: Click-bound options forwarded to
            [`run_model`][laser.cholera.metapop.model.run_model] as
            parameter mods. Notable keys: `seed` (int), `visualize` /
            `pdf` / `quiet` / `hdf5_output` / `compress` (bool), `outdir`
            (Path), `loglevel` (str — popped before forwarding), and
            `over` (tuple of `"key:value"` / `"key=value"` strings —
            parsed and type-coerced through
            [`override_helper`][laser.cholera.metapop.utils.override_helper]).

    Raises:
        click.UsageError: When `--over` references an unknown parameter
            name (with a `difflib` "did you mean" suggestion if a close
            match exists in the mapping).
        ValueError: When `--over` references a known but CLI-unsupported
            parameter (vector / matrix / DataFrame). Propagated as-is so
            the architectural problem stays visible.
    """

    logging.getLogger("laser.cholera").setLevel(kwargs.pop("loglevel", "INFO"))
    logger.info("Starting the cholera model simulation...")

    if overrides := kwargs.pop("over", ()):
        logger.info(f"Overriding parameters: {overrides}")
        parsed = {}
        for token in overrides:
            sep = "=" if "=" in token else ":"
            key, _, value = token.partition(sep)
            parsed[key] = value
        try:
            kwargs.update(override_helper(parsed))
        except UnknownOverrideKey as exc:
            raise click.UsageError(str(exc)) from exc

    run_model(params, **kwargs)

    return


def run_model(paramfile: Optional[Union[str, Path, dict]], **kwargs: Optional[dict]) -> Model:
    """Build and run the default cholera metapopulation simulation.

    The canonical Python entry point. Loads parameters via
    [`get_parameters`][laser.cholera.metapop.params.get_parameters],
    constructs a [`Model`][laser.cholera.metapop.model.Model] wired with
    the full default component pipeline (`Susceptible` → `Exposed` →
    `Recovered` → `Infectious` → `Vaccinated` → `Census` → `HumanToHuman`
    → `EnvToHuman` → `Environmental` → `DerivedValues` → `Analyzer` →
    `Recorder` → `Parameters`), runs the simulation to completion, and
    optionally renders visualizations.

    Args:
        paramfile: Parameter source — anything
            [`get_parameters`][laser.cholera.metapop.params.get_parameters]
            accepts. Use ``None`` for the bundled defaults, a filesystem
            path (``str`` / ``pathlib.Path``) to a JSON or JSON.gz file,
            or an in-memory ``dict``.
        **kwargs: Forwarded as ``mods`` to
            [`get_parameters`][laser.cholera.metapop.params.get_parameters]
            — same shape as the CLI ``--over key:value`` flags.

    Returns:
        The completed [`Model`][laser.cholera.metapop.model.Model] with
        per-tick state populated on ``model.people`` / ``model.patches``
        and the final-tick log-likelihood (when ``calc_likelihood=True``
        in params) on ``model.log_likelihood``. If ``params.visualize``
        or ``params.pdf`` is set, ``model.pdf`` is also populated with
        the rendered output path.

    Raises:
        ValueError: If ``paramfile`` is not one of the supported types
            (propagated from ``get_parameters``).

    Example:
        Run with defaults and inspect the final-tick susceptible counts:

        >>> from laser.cholera.metapop.model import run_model  # doctest: +SKIP
        >>> model = run_model(None)                            # doctest: +SKIP
        >>> model.people.S[-1].sum() > 0                       # doctest: +SKIP
        True
    """
    parameters = get_parameters(paramfile, mods=kwargs)

    model = Model(parameters)

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

    if parameters.visualize or parameters.pdf:
        model.pdf = model.visualize(pdf=parameters.pdf)

    return model


if __name__ == "__main__":
    ctx = click.Context(cli_run)
    ctx.invoke(cli_run, seed=20241107, loglevel="INFO", visualize=True, pdf=False, hdf5_output=False)
