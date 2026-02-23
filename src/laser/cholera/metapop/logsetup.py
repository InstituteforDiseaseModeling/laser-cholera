import logging
from datetime import datetime
from pathlib import Path

_log_file_handler = None


def setup_logging(loglevel, outdir: Path):
    global _log_file_handler

    if _log_file_handler is not None:
        return

    log_file_path = Path(outdir) / f"{datetime.now():%Y%m%d%H%M%S}.log"  # noqa: DTZ005
    logger = logging.getLogger()
    logger.setLevel(loglevel)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    class LazyFileHandler(logging.FileHandler):
        def __init__(self, filename, mode="a", encoding=None, delay=False):
            super().__init__(filename, mode, encoding, delay)
            self._filename = filename

        def emit(self, record):
            if not self.stream:
                self.stream = self._open()
            super().emit(record)

    _log_file_handler = LazyFileHandler(log_file_path, mode="a", encoding="utf-8", delay=True)
    _log_file_handler.setFormatter(formatter)
    logger.addHandler(_log_file_handler)

    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(formatter)
    # logger.addHandler(console_handler)

    return


setup_logging("WARNING", Path.cwd())
# setup_logging("DEBUG", Path.cwd())
