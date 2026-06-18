import unittest

import click
import pytest

from laser.cholera.metapop.model import cli_run


class TestCholeraMPM(unittest.TestCase):
    def test_cholera_metapopulation_model(self):
        # Run the metapopulation model with default parameters
        ctx = click.Context(cli_run)
        ctx.invoke(cli_run, seed=20250326, loglevel="WARNING", viz=False, pdf=False)


class TestCliOverrideValidation(unittest.TestCase):
    """CLI-boundary tests for `--over` key validation.

    `cli_run` re-raises `UnknownOverrideKey` (from `override_helper`) as
    `click.UsageError` so CLI users get a clean message instead of a raw
    traceback. Architectural rejects (vectors / matrices passed via
    `--over`) propagate as plain `ValueError` because they indicate a
    misuse of the override mechanism, not a typo.
    """

    def test_unknown_over_key_raises_usage_error_with_suggestion(self):
        """A misspelled `--over` key reaches the user as a click.UsageError.

        Given a `--over` tuple containing a near-miss key (``date_strat``),
        when `cli_run` parses it,
        then `override_helper` raises `UnknownOverrideKey`, which `cli_run`
        catches and re-raises as `click.UsageError` whose message names
        the typo and suggests the correct spelling.

        Failure implies the `UnknownOverrideKey` → `UsageError` bridge in
        `cli_run` has regressed; CLI users would see a stack trace instead
        of a one-line "Usage:" hint.
        """
        ctx = click.Context(cli_run)
        with pytest.raises(click.UsageError, match=r"date_strat.*date_start") as exc_info:
            ctx.invoke(cli_run, seed=20250326, loglevel="WARNING", viz=False, pdf=False, over=("date_strat:2024-01-01",))
        message = str(exc_info.value)
        assert "date_strat" in message
        assert "date_start" in message

    def test_cli_unsupported_over_key_raises_value_error(self):
        """Vector / matrix overrides via --over surface as raw ValueError.

        Given a `--over` tuple containing a known but CLI-unsupported key
        (``b_jt`` — a matrix),
        when `cli_run` parses it,
        then `override_helper` raises plain `ValueError` (NOT
        `UnknownOverrideKey`) and `cli_run` does NOT re-wrap it as
        `click.UsageError`; the architectural problem stays visible.

        Failure implies either the `_cli_unsupported` factory regressed
        (silently forwarding a string to a matrix slot) or `cli_run`'s
        narrow catch is now over-catching and downgrading architectural
        errors to user-input errors.
        """
        ctx = click.Context(cli_run)
        with pytest.raises(ValueError, match=r"b_jt.*cannot be set via --over") as exc_info:
            ctx.invoke(cli_run, seed=20250326, loglevel="WARNING", viz=False, pdf=False, over=("b_jt:anything",))
        # Must NOT be a UsageError — that path is reserved for typos.
        assert not isinstance(exc_info.value, click.UsageError)


if __name__ == "__main__":
    unittest.main()
