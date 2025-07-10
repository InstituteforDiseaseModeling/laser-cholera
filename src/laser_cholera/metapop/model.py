import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from laser_core.laserframe import LaserFrame
from laser_core.propertyset import PropertySet
from laser_core.random import seed as seed_prng
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from tqdm import tqdm

from laser_cholera.metapop import Analyzer
from laser_cholera.metapop import Census
from laser_cholera.metapop import DerivedValues
from laser_cholera.metapop import Environmental
from laser_cholera.metapop import EnvToHuman
from laser_cholera.metapop import EnvToHumanVax
from laser_cholera.metapop import Exposed
from laser_cholera.metapop import HumanToHuman
from laser_cholera.metapop import HumanToHumanVax
from laser_cholera.metapop import Infectious
from laser_cholera.metapop import Parameters
from laser_cholera.metapop import Recorder
from laser_cholera.metapop import Recovered
from laser_cholera.metapop import Susceptible
from laser_cholera.metapop import Vaccinated
from laser_cholera.metapop import get_parameters
from laser_cholera.metapop import scenario
from laser_cholera.metapop.utils import override_helper

logger = logging.getLogger(__name__)


class Model:
    def __init__(self, parameters: PropertySet, name: str = "Cholera Metapop"):
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

    def plot(self, fig: Figure = None):  # pragma: no cover
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
@click.option("--seed", default=20241107, help="Random seed")
@click.option("--viz", "visualize", is_flag=True, default=False, help="Display visualizations")
@click.option("--pdf", is_flag=True, default=False, help="Output visualization results as a PDF")
@click.option("--outdir", "-o", default=Path.cwd(), help="Output file for results")
@click.option("--params", "-p", default=None, help="JSON file with parameters")
@click.option("--over", multiple=True, help="Additional parameter overrides (param:value or param=value)")
@click.option("--loglevel", default="WARNING", help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
@click.option("-q", "--quiet", is_flag=True, default=False, help="Suppress console progress output")
def cli_run(params, **kwargs):
    """
    Run the cholera model simulation with the given parameters.

    This function initializes the model with the specified parameters, sets up the
    components of the model, seeds initial infections, runs the simulation, and
    optionally visualizes the results.

    Parameters:

        **kwargs: Arbitrary keyword arguments containing the parameters for the simulation.

            Expected keys include:

                - "loglevel": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] logging level.
                - "viz": (bool) Whether to show visualizations.
                - "pdf": (str) The file path to save the visualization as a PDF.

    Returns:

        None
    """

    # logger.setLevel(kwargs.pop("loglevel", "INFO"))
    logging.getLogger().setLevel(kwargs.pop("loglevel", "INFO"))  # Set the root logger level
    logger.info("Starting the cholera model simulation...")

    if "over" in kwargs and (overrides := kwargs.pop("over")):
        logger.info(f"Overriding parameters: {overrides}")
        for override in overrides:
            param, value = override.split("=") if "=" in override else override.split(":")
            kwargs[param] = value
        typed = override_helper(kwargs)
        kwargs.update(typed)

    run_model(params, **kwargs)

    return


def run_model(paramfile, **kwargs):
    parameters = get_parameters(paramfile, overrides=kwargs)

    model = Model(parameters)

    model.components = [
        Susceptible,
        Exposed,
        Recovered,
        Infectious,
        Vaccinated,
        Census,
        HumanToHuman,
        HumanToHumanVax,
        EnvToHuman,
        EnvToHumanVax,
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
    ctx.invoke(cli_run, seed=20241107, loglevel="INFO", visualize=True, pdf=False, over=["hdf5_output:0"])
