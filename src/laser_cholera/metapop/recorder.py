import gzip
import io
import logging
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Union

import h5py as h5

logger = logging.getLogger(__name__)


class Recorder:
    def __init__(self, model) -> None:
        self.model = model
        return

    def check(self):
        if not hasattr(self.model, "people"):
            Warning("Recorder: model expected to have a 'people' attribute.")
        if not hasattr(self.model, "patches"):
            Warning("Recorder: model expected to have a 'patches' attribute.")

        return

    def __call__(self, model, tick: int) -> None:
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


def save_hdf5_parameters(model, filename: Union[str, Path]) -> Path:
    # Step 1: Write HDF5 content to the file
    with h5.File(filename, "w") as h5file:
        save_hdf5(h5file, model)

    return filename  # Unmodified


def save_compressed_hdf5_parameters(model, filename: Union[str, Path]) -> Path:
    # Step 1: Create an in-memory buffer for HDF5
    hdf5_buffer = io.BytesIO()

    # Step 2: Write HDF5 content to the buffer
    with h5.File(hdf5_buffer, "w") as h5file:
        save_hdf5(h5file, model)

    # Step 3: Compress and save to disk
    filename = filename.with_name(filename.name + ".gz")
    with gzip.open(filename, "wb") as gz_file:
        gz_file.write(hdf5_buffer.getvalue())

    return filename  # With .gz extension


def save_hdf5(h5file: h5.File, model) -> None:
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
