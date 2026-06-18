"""Process-wide logging setup for the `laser.cholera` package.

Installs a lazy file handler (the file is only opened on first write,
so a run that never logs leaves no trash log behind) on the
`laser.cholera` logger. Idempotent: subsequent calls are no-ops, so it
is safe to import multiple times and to be re-invoked from `cli_run`.

Imported and called once at import time with a `WARNING` level so the
log file path is established before any other component imports
`logger`.
"""

import logging
from datetime import datetime
from pathlib import Path

_log_file_handler = None


def setup_logging(loglevel: str | int, outdir: Path) -> None:
    """Configure the `laser.cholera` logger with a lazy timestamped file handler.

    The handler is `LazyFileHandler` (subclasses `logging.FileHandler`
    with `delay=True` and opens the file only on first emit) — runs
    that never log do not leave an empty log file. Idempotent: after
    the first call the global `_log_file_handler` is set, and further
    calls return immediately without re-configuring the logger.

    Args:
        loglevel: Logging level to apply to the `laser.cholera` logger.
            Accepts a string (`"DEBUG"`, `"INFO"`, …) or an int.
        outdir: Directory the log file is created in. The filename is
            derived from the current wall-clock time
            (`%Y%m%d%H%M%S.log`).
    """
    global _log_file_handler

    if _log_file_handler is not None:
        return

    log_file_path = Path(outdir) / f"{datetime.now():%Y%m%d%H%M%S}.log"  # noqa: DTZ005
    logger = logging.getLogger("laser.cholera")
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
