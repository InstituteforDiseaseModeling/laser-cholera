"""Final-tick HDF5 dump of the simulation state.

[`Recorder`][laser.cholera.metapop.recorder.Recorder] writes selected
`people` / `patches` properties to an HDF5 file on the last tick of
the run, gated by two parameter flags:

- `hdf5_output` (bool) — master enable.
- `return` (list[str]) — whitelist of property names to dump
  (everything not in this list is skipped).

Optionally writes gzip-compressed output when `compress=True`. Output
goes into `params.outdir` (or `cwd` if absent), named with the current
wall-clock timestamp (`%Y%m%d%H%M%S.h5` or `.h5.gz`).

The low-level
[`save_hdf5`][laser.cholera.metapop.recorder.save_hdf5] helper walks
each `LaserFrame`, filters to public, non-method, whitelisted
properties, and writes each as an HDF5 dataset inside a per-frame
group.
"""

import gzip
import io
import logging
import warnings
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import TYPE_CHECKING

import h5py as h5

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model

logger = logging.getLogger("laser.cholera")


class Recorder:
    """Pipeline component that dumps a configurable HDF5 snapshot on the final tick.

    No per-tick work until the last tick; gated by `params.hdf5_output`
    and the `params.return` property whitelist.

    Attributes:
        model: The parent `Model` instance.
    """

    def __init__(self, model: "Model") -> None:
        """Register the component on `model` (no state is allocated).

        Args:
            model: The `Model` instance.
        """
        self.model = model
        return

    def check(self):
        """Warn (but do not fail) if expected `people` / `patches` frames are missing.

        Uses `warnings.warn` rather than `assert` so a recorder
        configured against an unusual minimal model still lets the
        run complete; the final-tick dump will then simply have no
        groups to write.
        """
        if not hasattr(self.model, "people"):
            warnings.warn("Recorder: model expected to have a 'people' attribute.", stacklevel=1)
        if not hasattr(self.model, "patches"):
            warnings.warn("Recorder: model expected to have a 'patches' attribute.", stacklevel=1)

        return

    def __call__(self, model: "Model", tick: int) -> None:
        """On the final tick, write whitelisted state to HDF5 if `hdf5_output` is set.

        Quietly returns (with an info-level log explaining which gate
        was unmet) when either `params.hdf5_output` is false or
        `params.return` is unset / empty. When both are present and
        truthy, dispatches to
        [`save_hdf5_parameters`][laser.cholera.metapop.recorder.save_hdf5_parameters]
        or
        [`save_compressed_hdf5_parameters`][laser.cholera.metapop.recorder.save_compressed_hdf5_parameters]
        based on `params.compress`.

        Args:
            model: The parent `Model` instance.
            tick: Current simulation tick.
        """
        # write the current state of the model to an HDF5 file if it is the last tick of the simulation
        if tick == (model.params.nticks - 1):
            # To get output must a) specify hdf5_output = true in the params file and b) specify properties to return
            write_output = ("hdf5_output" in model.params) and model.params.hdf5_output and ("return" in model.params) and model.params["return"]

            if write_output:
                root = Path(model.params.outdir) if "outdir" in model.params and model.params.outdir else Path.cwd()
                root.mkdir(parents=True, exist_ok=True)
                filename = root / f"{datetime.now().strftime('%Y%m%d%H%M%S')}.h5"  # noqa: DTZ005

                if ("compress" in model.params) and model.params.compress:
                    filename = save_compressed_hdf5_parameters(model, filename)
                else:
                    filename = save_hdf5_parameters(model, filename)

                logger.info(f"Recorder: model state saved to {filename}")
            else:
                logger.info("Recorder: model state not saved to HDF5 file.")
                logger.info(
                    f"\t'hdf5_output' {'is' if 'hdf5_output' in model.params else 'is not'} in the params file."
                    + (f" hdf5_output = {model.params.hdf5_output}" if "hdf5_output" in model.params else "")
                )
                logger.info(
                    f"\t'return' {'is' if 'return' in model.params else 'is not'} in the params file."
                    + (f" return = {model.params['return']}" if "return" in model.params else "")
                )

        return

    # def plot(self, fig: Figure = None):  # pragma: no cover
    #     yield
    #     return


def save_hdf5_parameters(model: "Model", filename: str | Path) -> Path:
    """Write a model's whitelisted `people` / `patches` properties to an HDF5 file.

    Args:
        model: The `Model` instance whose `people` and `patches` frames
            are dumped.
        filename: Destination path (str or Path). Created or
            overwritten.

    Returns:
        The destination path as a `Path` (unchanged).
    """
    # Step 1: Write HDF5 content to the file
    with h5.File(filename, "w") as h5file:
        save_hdf5(h5file, model)

    return Path(filename)  # Unmodified


def save_compressed_hdf5_parameters(model: "Model", filename: str | Path) -> Path:
    """Write the model state to an in-memory HDF5 buffer, then gzip it to disk.

    The returned filename has a `.gz` suffix appended to the input
    (so `foo.h5` becomes `foo.h5.gz`).

    Args:
        model: The `Model` instance.
        filename: Base destination path (no `.gz` required; one is
            appended).

    Returns:
        The destination path with `.gz` appended.
    """
    # Step 1: Create an in-memory buffer for HDF5
    hdf5_buffer = io.BytesIO()

    # Step 2: Write HDF5 content to the buffer
    with h5.File(hdf5_buffer, "w") as h5file:
        save_hdf5(h5file, model)

    # Step 3: Compress and save to disk
    filename = Path(filename)
    filename = filename.with_name(filename.name + ".gz")
    with gzip.open(filename, "wb") as gz_file:
        gz_file.write(hdf5_buffer.getvalue())

    return filename  # With .gz extension


def save_hdf5(h5file: h5.File, model: "Model") -> None:
    """Write the whitelisted properties of `model.people` and `model.patches` to an open HDF5 file.

    For each of the two `LaserFrame` instances on `model`, enumerates
    `dir(frame)`, filters out `_`-prefixed attributes and bound
    methods, intersects with `model.params["return"]`, and writes each
    remaining property as a dataset inside a per-frame HDF5 group
    (`/people/<prop>`, `/patches/<prop>`).

    Args:
        h5file: An open `h5py.File` opened in write mode.
        model: The `Model` instance.

    Raises:
        AttributeError: When `model` is missing the `people` or
            `patches` frame.
    """
    for frame in ["people", "patches"]:
        if not hasattr(model, frame):
            raise AttributeError(f"Recorder: model needs to have a '{frame}' attribute.")

        lf = getattr(model, frame)
        properties = dir(lf)

        # TODO - automagically write all properties if a subset is not specified

        properties = [p for p in properties if not p.startswith("_")]
        properties = [p for p in properties if not isinstance(getattr(lf, p), (MethodType))]

        # TODO - if a subset is specified, don't bother with the filtering above

        properties = [p for p in properties if p in model.params["return"]]

        group = h5file.create_group(frame)

        for prop in properties:
            logger.info(f"Recorder: saving {frame}.{prop} ...")
            group.create_dataset(prop, data=getattr(lf, prop))

    return
