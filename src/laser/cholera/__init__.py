"""`laser.cholera` — LASER-based metapopulation cholera simulation.

Top-level package for LASIK (LASER-cholera). The bulk of the simulation
lives in the [`metapop`][laser.cholera.metapop] subpackage; this
module's main job is to expose the package version and a couple of
convenience re-exports.

Exports:
    __version__: Installed package version (read from package
        metadata).
    compute: Sample compute kernel from [`laser.cholera.core`].
    iso_codes: Tuple of ISO-3 country codes covered by the bundled
        MOSAIC scenario; see
        [`laser.cholera.iso_codes`][laser.cholera.iso_codes].
"""

from importlib.metadata import version

__version__ = version("laser.cholera")

from .core import compute
from .iso_codes import iso_codes

__all__ = [
    "__version__",
    "compute",
    "iso_codes",
]
