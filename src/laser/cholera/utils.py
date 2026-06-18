"""Top-level utilities for `laser.cholera`.

Currently a single helper,
[`sim_duration`][laser.cholera.utils.sim_duration], for building a
parameter-override dict that shrinks the simulation window to a
requested calendar interval. Used by tests and short-run scripts to
short-cut the bundled `default_parameters.json`'s ~1155-day default
window. NB: callers also need to slice the time-series matrices
(`b_jt`, `d_jt`, `nu_1_jt`, `nu_2_jt`, `psi_jt`) to match the new
`nticks`; see `tests/test_model.py` for the pattern.
"""

from datetime import datetime


def sim_duration(start: datetime = datetime(2025, 3, 24), stop: datetime = datetime(2025, 4, 24)) -> dict:
    """Build a parameter-override dict that shrinks the simulation window.

    Returns the minimum set of fields needed to short-cut a simulation to
    the given calendar interval: ``date_start``, ``date_stop`` (formatted
    as ``YYYY-MM-DD`` strings), and ``nticks`` (the inclusive day count).
    Suitable as the ``mods=`` argument to
    [`get_parameters`][laser.cholera.metapop.params.get_parameters].

    Args:
        start: Calendar start date of the run. Defaults to 2025-03-24.
        stop: Calendar stop date of the run, inclusive. Defaults to
            2025-04-24.

    Returns:
        A dict with keys ``date_start`` (ISO-format str), ``date_stop``
        (ISO-format str), and ``nticks`` (int = stop − start + 1).

    Example:
        >>> from datetime import datetime
        >>> from laser.cholera.utils import sim_duration
        >>> sim_duration(datetime(2024, 1, 1), datetime(2024, 1, 31))
        {'date_start': '2024-01-01', 'date_stop': '2024-01-31', 'nticks': 31}
    """
    return {"date_start": f"{start:%Y-%m-%d}", "date_stop": f"{stop:%Y-%m-%d}", "nticks": (stop - start).days + 1}
