from importlib.metadata import version

__version__ = version("laser.cholera")

from .core import compute
from .iso_codes import iso_codes

__all__ = [
    "__version__",
    "compute",
    "iso_codes",
]
