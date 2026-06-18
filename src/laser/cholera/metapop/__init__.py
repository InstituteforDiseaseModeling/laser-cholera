"""`laser.cholera.metapop` — metapopulation cholera simulation pipeline.

Re-exports the canonical component classes and `get_parameters` /
`scenario` helpers as a flat namespace, so user code can write
`from laser.cholera.metapop import Susceptible, Exposed, ...` instead
of importing from each submodule. Importing this package also
triggers [`logsetup`][laser.cholera.metapop.logsetup] which configures
the `laser.cholera` logger and reserves a timestamped log file path.

The default component pipeline (`Susceptible` -> `Exposed` ->
`Recovered` -> `Infectious` -> `Vaccinated` -> `Census` ->
`HumanToHuman` -> `EnvToHuman` -> `Environmental` -> `DerivedValues` ->
`Analyzer` -> `Recorder` -> `Parameters`) is wired up by
[`run_model`][laser.cholera.metapop.model.run_model].
"""

from . import logsetup  # noqa: F401, I001


from .analyzer import Analyzer
from .census import Census
from .derivedvalues import DerivedValues
from .environmental import Environmental
from .envtohuman import EnvToHuman
from .exposed import Exposed
from .humantohuman import HumanToHuman
from .infectious import Infectious
from .params import Parameters
from .params import get_parameters
from .recorder import Recorder
from .recovered import Recovered
from .scenario import scenario
from .susceptible import Susceptible
from .vaccinated import Vaccinated

__all__ = [
    "Analyzer",
    "Census",
    "DerivedValues",
    "EnvToHuman",
    "Environmental",
    "Exposed",
    "HumanToHuman",
    "Infectious",
    "Parameters",
    "Recorder",
    "Recovered",
    "Susceptible",
    "Vaccinated",
    "get_parameters",
    "scenario",
]
