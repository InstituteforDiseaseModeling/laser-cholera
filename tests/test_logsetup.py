"""Tests for `laser.cholera.metapop.logsetup.setup_logging`.

`setup_logging` is invoked once at package import time (with
`loglevel="WARNING"`), which means by the time any test runs, the
module-level `_log_file_handler` is already populated and the
`laser.cholera` logger already carries a `LazyFileHandler`. The tests
therefore reset that module global and detach the import-time handler
in `setUp`, and restore both in `tearDown`, so each test exercises a
fresh configuration of the function under test without leaking state
into the rest of the suite.
"""

import logging
import tempfile
import unittest
from pathlib import Path

from laser.cholera.metapop import logsetup


class TestSetupLogging(unittest.TestCase):
    """Tests for `setup_logging` covering both `str` and `int` loglevel inputs."""

    def setUp(self):
        """Snapshot logsetup module state and reset to a "never-called" baseline.

        Records the existing `_log_file_handler` (set by the import-time
        invocation) plus the `laser.cholera` logger's level and handler
        list, then removes any `LazyFileHandler` instances and nulls the
        module global so `setup_logging` will execute its configuration
        body rather than short-circuit on the idempotency guard.
        """
        self._original_handler = logsetup._log_file_handler
        self._logger = logging.getLogger("laser.cholera")
        self._original_level = self._logger.level
        self._original_handlers = list(self._logger.handlers)

        # Detach any LazyFileHandler the import-time call attached so the
        # post-test handler-count assertions are unambiguous.
        for handler in list(self._logger.handlers):
            if handler.__class__.__name__ == "LazyFileHandler":
                self._logger.removeHandler(handler)
        logsetup._log_file_handler = None

    def tearDown(self):
        """Restore the original `_log_file_handler` and logger state.

        Without this, the next test (or the rest of the suite) would see
        whatever level / handler the current test installed, defeating
        isolation.
        """
        # Strip out anything this test attached.
        for handler in list(self._logger.handlers):
            if handler not in self._original_handlers:
                self._logger.removeHandler(handler)
        # Restore originals.
        for handler in self._original_handlers:
            if handler not in self._logger.handlers:
                self._logger.addHandler(handler)
        self._logger.setLevel(self._original_level)
        logsetup._log_file_handler = self._original_handler

    def test_setup_logging_accepts_string_loglevel(self):
        """A string loglevel like `"DEBUG"` is applied to the `laser.cholera` logger.

        Given the module-level `_log_file_handler` has been reset and the
        `laser.cholera` logger has no `LazyFileHandler` attached,
        when `setup_logging("DEBUG", <tmpdir>)` is called,
        then `_log_file_handler` is populated with a `LazyFileHandler`
        instance, the logger's level is `logging.DEBUG` (10), the handler
        is attached to the logger, and — because the file handler is
        lazy — no log file has been created on disk yet.

        Failure implies the string-loglevel path is broken (e.g.,
        `setLevel` no longer accepts strings) or the `LazyFileHandler`
        wiring has regressed.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            logsetup.setup_logging("DEBUG", outdir)

            handler = logsetup._log_file_handler
            assert handler is not None, "setup_logging should have populated _log_file_handler"
            assert handler.__class__.__name__ == "LazyFileHandler", f"expected LazyFileHandler, got {type(handler).__name__}"
            assert self._logger.level == logging.DEBUG, f"expected logger level DEBUG (10), got {self._logger.level}"
            assert handler in self._logger.handlers, "the LazyFileHandler should be attached to the laser.cholera logger"
            # Laziness: the log file path is reserved but the file itself
            # is not created until the first record is emitted.
            assert not any(outdir.iterdir()), f"no log file should be created until a record is emitted; found {list(outdir.iterdir())}"

    def test_setup_logging_accepts_integer_loglevel(self):
        """An integer loglevel like `logging.INFO` (20) is applied to the logger.

        Given the module-level `_log_file_handler` has been reset,
        when `setup_logging(logging.INFO, <tmpdir>)` is called (passing
        the int `20`),
        then `_log_file_handler` is populated and the logger's level is
        the integer that was passed in.

        Failure implies the integer-loglevel path is broken (e.g., a
        future refactor mistakenly added an `isinstance(..., str)` gate
        on the level argument).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir)

            logsetup.setup_logging(logging.INFO, outdir)

            handler = logsetup._log_file_handler
            assert handler is not None, "setup_logging should have populated _log_file_handler"
            assert self._logger.level == logging.INFO, f"expected logger level INFO (20), got {self._logger.level}"
            assert self._logger.level == 20, "logging.INFO should be the integer 20 — sanity check on the constant"


if __name__ == "__main__":
    unittest.main()
