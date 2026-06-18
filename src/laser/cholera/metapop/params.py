"""Parameter loading, typing, validation, and the `Parameters` reporting component.

Three layered jobs:

1. **I/O.** [`load_json_parameters`][laser.cholera.metapop.params.load_json_parameters]
   and
   [`load_compressed_json_parameters`][laser.cholera.metapop.params.load_compressed_json_parameters]
   read a JSON / JSON.gz file into a plain dict.
2. **Typing & coercion.**
   [`dict_to_propertysetex`][laser.cholera.metapop.params.dict_to_propertysetex]
   wraps the dict in a [`PropertySetEx`][laser.cholera.metapop.params.PropertySetEx]
   and coerces ~30 scalars to `np.float32` / `np.int32` and ~40 vectors /
   matrices to `np.ndarray` of the appropriate dtype; converts ISO date
   strings to `datetime`; promotes `epidemic_peaks` to a DataFrame with
   a `loc_idx` column. The
   [`as_ndarray`][laser.cholera.metapop.params.as_ndarray] and
   [`handle_nan`][laser.cholera.metapop.params.handle_nan] helpers live
   here.
3. **Validation.**
   [`validate_parameters`][laser.cholera.metapop.params.validate_parameters]
   enforces shape and range invariants over the typed result.

The
[`get_parameters`][laser.cholera.metapop.params.get_parameters] facade
composes those steps and applies any caller-supplied `mods` overrides.

[`Parameters`][laser.cholera.metapop.params.Parameters] is a pipeline
component that owns no per-tick logic but renders an extensive
parameter-overview PDF section via its
[`plot`][laser.cholera.metapop.params.Parameters.plot] generator
(initial populations, birth / mortality / vaccination / disease-mortality
rate heatmaps, emigration / WASH scatter, suitability heatmap).

A small custom JSON encoder,
[`PseEncoder`][laser.cholera.metapop.params.PseEncoder], teaches
`json.dumps` how to serialize `PropertySet` / `np.ndarray` / `datetime`
/ `np.integer` / `np.floating` / `pd.DataFrame` values for `__str__` on
a `PropertySetEx`.
"""

import gzip
import io
import json
import logging
from collections.abc import Iterator
from datetime import datetime
from numbers import Number
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from laser.core.propertyset import PropertySet
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from laser.cholera.metapop.model import Model
logger = logging.getLogger("laser.cholera")


class PseEncoder(json.JSONEncoder):
    """JSON encoder that knows how to serialize the types stored on a `PropertySetEx`.

    Recognized: `PropertySet` (via `.to_dict()`), `np.ndarray`
    (`.tolist()`), `datetime` (ISO 8601 `YYYY-MM-DD`), `np.integer`,
    `np.floating`, and `pd.DataFrame` (records-orient list). Anything
    else falls through to `JSONEncoder.default` (which raises
    `TypeError` for unsupported types). Used by
    [`PropertySetEx.__str__`][laser.cholera.metapop.params.PropertySetEx]
    to produce a human-readable JSON dump.
    """

    def default(self, o: object) -> object:
        """Convert one non-stdlib-JSON-serializable value to a JSON-friendly form.

        Args:
            o: The value `json` is trying to serialize.

        Returns:
            A JSON-compatible representation (`dict`, `list`, `str`,
            `int`, or `float`).

        Raises:
            TypeError: For any type not in the recognized list,
                propagated from `JSONEncoder.default`.
        """
        if isinstance(o, PropertySet):
            return o.to_dict()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        elif isinstance(o, datetime):
            return f"{o:%Y-%m-%d}"
        elif isinstance(o, np.integer):
            return int(o)
        elif isinstance(o, np.floating):
            return float(o)
        elif isinstance(o, pd.DataFrame):
            return o.to_dict(orient="records")
        else:
            return super().default(o)


class PropertySetEx(PropertySet):
    """`PropertySet` with a richer `__str__` that round-trips through `PseEncoder`.

    Adds no new state — only a custom string representation that
    formats the contained dict with indented JSON and the encoder's
    type-aware conversions. Inherits all of `PropertySet`'s
    attribute-access, `<<=` override, and `+=` extend semantics.
    """

    def __init__(self, *kvps: object) -> None:
        """Forward all positional args to `PropertySet.__init__`.

        Args:
            *kvps: One or more key/value mappings or `(k, v)` pairs;
                the canonical form is a single `dict`.
        """
        super().__init__(*kvps)

        return

    def __str__(self) -> str:
        """Return a string representation of the PropertySet.

        Include converters for datetime and NumPy types.
        """
        return json.dumps(self.to_dict(), cls=PseEncoder, indent=4)


def get_parameters(
    paramsource: str | Path | dict | None = None,
    do_validation: bool = True,
    mods: dict | None = None,
) -> PropertySetEx:
    """Load parameters from disk or memory and return a typed ``PropertySetEx``.

    The canonical entry point for assembling a simulation parameter set.
    Accepts a parameter source in any of four forms:

    - ``None`` → loads ``src/laser/cholera/metapop/data/default_parameters.json``
      (the bundled defaults).
    - ``str`` or ``pathlib.Path`` → loaded from that filesystem path.
      Supported suffixes are ``.json`` and ``.json.gz``; HDF5 sources are
      no longer supported.
    - ``dict`` → ingested in-memory via
      [`dict_to_propertysetex`][laser.cholera.metapop.params.dict_to_propertysetex].

    Any of those is then optionally merged with caller-supplied ``mods``:
    overrides of existing keys are applied first (logged at INFO level),
    additions of new keys follow. After merge, ``validate_parameters`` is
    called unless ``do_validation=False``. Finally, default values for
    the visualization/output flags (``visualize``, ``pdf``, ``quiet``)
    are filled in if absent so downstream code can read them
    unconditionally.

    Args:
        paramsource: ``None`` for the bundled defaults, a filesystem
            path (str or Path) to a JSON/JSON.gz file, or an already-
            built parameter dict.
        do_validation: When True (default), run
            [`validate_parameters`][laser.cholera.metapop.params.validate_parameters]
            on the assembled result. Set to ``False`` for tests that
            deliberately construct partial or invariant-violating
            configurations.
        mods: Optional dict of overrides/additions to merge after loading
            the base parameters. Typed via
            [`override_helper`][laser.cholera.metapop.utils.override_helper]
            on the CLI path; passed verbatim from Python callers.

    Returns:
        A populated [`PropertySetEx`][laser.cholera.metapop.params.PropertySetEx]
        with all known fields coerced to their working dtypes and
        (when ``do_validation=True``) verified against the model's
        invariants.

    Raises:
        KeyError: If ``paramsource`` is a path whose suffix is not in the
            dispatch table (e.g., a ``.h5`` file after the HDF5 ingest
            path was removed).
        ValueError: If ``paramsource`` is not None, str, Path, or dict —
            including the historically-supported HDF5 paths.
        ValueError: From ``validate_parameters`` if the resulting
            parameter set violates the model's invariants and
            ``do_validation=True``.

    Example:
        Load the bundled defaults with a one-month simulation window:

        >>> from laser.cholera.metapop.params import get_parameters
        >>> from laser.cholera.utils import sim_duration
        >>> from datetime import datetime
        >>> params = get_parameters(mods=sim_duration(datetime(2024, 1, 1), datetime(2024, 1, 31)))
        >>> params.nticks
        31
    """
    fn_map = {
        (".json",): load_json_parameters,
        (".json", ".gz"): load_compressed_json_parameters,
    }

    if isinstance(paramsource, (str, Path, type(None))):
        file_path = Path(paramsource) if paramsource is not None else Path(__file__).parent / "data" / "default_parameters.json"
        suffixes = [suffix.lower() for suffix in file_path.suffixes]
        load_fn = fn_map[tuple(suffixes)]

        logger.info(f"Loading parameters from `{file_path}`…")
        params = load_fn(file_path)

    elif isinstance(paramsource, dict):
        params = dict_to_propertysetex(paramsource)

    else:
        raise ValueError(f"Invalid parameter source type: {type(paramsource)}")

    if mods is not None:
        overrides = {k: v for k, v in mods.items() if k in params}
        additions = {k: v for k, v in mods.items() if k not in params}

        if overrides:
            # Update the parameters with the overrides
            params <<= overrides

            logger.info("Updated/overrode file parameters with overrides:")
            for k, v in overrides.items():
                logger.info(f"  '{k}': {v}")

        if additions:
            # Add the additional parameters
            params += additions

            logger.info("Parameters added to file parameters:")
            for k, v in additions.items():
                logger.info(f"  '{k}': {v}")

    if do_validation:
        validate_parameters(params)

    if "visualize" not in params:
        params.visualize = False
    if "pdf" not in params:
        params.pdf = False
    if "quiet" not in params:
        params.quiet = True

    return params


def load_json_parameters(filename: str | Path) -> PropertySetEx:
    """Read a plain JSON file and return a typed `PropertySetEx`.

    Wraps `json.load` + `dict_to_propertysetex`. Used by
    `get_parameters` when the suffix tuple is `(".json",)`.

    Args:
        filename: Path to a `.json` file containing a parameters
            object.

    Returns:
        A typed `PropertySetEx` with every recognized field coerced to
        its runtime dtype.
    """
    file_path = Path(filename)
    with file_path.open("r") as file:
        parameters = json.load(file)

    return dict_to_propertysetex(parameters)


def load_compressed_json_parameters(filename: str | Path) -> PropertySetEx:
    """Read a gzip-compressed JSON file and return a typed `PropertySetEx`.

    Used by `get_parameters` when the suffix tuple is `(".json", ".gz")`.

    Args:
        filename: Path to a `.json.gz` file.

    Returns:
        A typed `PropertySetEx`.
    """
    file_path = Path(filename)
    with gzip.open(file_path, "rb") as gz_file:
        with io.BytesIO(gz_file.read()) as file:
            parameters = json.load(file)

    return dict_to_propertysetex(parameters)


def as_ndarray(input: object, dtype: type) -> np.ndarray:
    """Coerce a list / scalar / existing ndarray to an `np.ndarray` of `dtype`.

    Pass-through for an existing `np.ndarray` (no extra copy). For
    lists, replaces any `"NA"` sentinel string with `0` before
    constructing the array (legacy behavior — upstream MOSAIC R data
    sometimes uses the R `NA` literal). Scalars are promoted to a
    one-element array.

    Args:
        input: An `np.ndarray`, a `list` (possibly containing `"NA"`),
            or a scalar.
        dtype: Target NumPy dtype.

    Returns:
        An `np.ndarray` of `dtype` containing the converted values.
    """
    retval = None

    if isinstance(input, np.ndarray):
        # Don't make yet another NumPy array ...
        retval = input
    elif isinstance(input, list):
        # Convert lists to NumPy artrays
        sanitized = [value if value != "NA" else 0 for value in input]
        retval = np.array(sanitized, dtype=dtype)
    else:
        # Convert other (assumed to be scalar) to a single entry NumPy array
        retval = np.array([input], dtype=dtype)

    return retval


def handle_nan(values: object, dtype: type) -> np.ndarray:
    """Convert a list-of-lists with mixed numeric / non-numeric cells to an `np.ndarray`.

    Each cell is run through `int(cell)`; cells that fail conversion
    become `np.nan` (which is why the caller passes a floating dtype
    — integer dtypes cannot hold NaN). Used to ingest the
    `reported_cases` / `reported_deaths` matrices, where missing weeks
    arrive as non-numeric placeholders.

    Args:
        values: A list of lists (rows × columns of mixed-type cells)
            or an existing `np.ndarray` (pass-through).
        dtype: Target NumPy dtype. Must be a floating type if the
            input contains any non-numeric cells.

    Returns:
        An `np.ndarray` of `dtype`. Non-numeric input cells appear as
        `np.nan`.

    Raises:
        AssertionError: When `values` is neither a list nor an
            `np.ndarray`.
    """

    def convert(item):
        try:
            return int(item)
        except ValueError:
            pass
        return np.nan

    if isinstance(values, list):
        values = np.array([[convert(element) for element in row] for row in values], dtype=dtype)
    else:
        assert isinstance(values, np.ndarray)

    return values


def dict_to_propertysetex(parameters: dict) -> PropertySetEx:
    """Wrap a parameters dict in a ``PropertySetEx`` and coerce every field to its runtime dtype.

    The canonical "raw dict → typed parameter set" converter. Performs all
    of the following in one pass:

    - Wraps the input dict in a [`PropertySetEx`][laser.cholera.metapop.params.PropertySetEx]
      (attribute-access view over the same key/value pairs).
    - Parses ``date_start`` / ``date_stop`` ISO-format strings into
      ``datetime`` instances (if they aren't already typed).
    - Computes ``nticks`` from the calendar window
      (``(date_stop - date_start).days + 1``).
    - Promotes scalar single-location ``location_name`` values to a
      one-element list so downstream code can always iterate.
    - Coerces ~30 scalar fields to ``np.float32`` / ``np.int32`` and ~40
      vector / matrix fields to ``np.ndarray`` of the appropriate dtype.
    - Transposes ``b_jt``, ``d_jt``, ``nu_1_jt``, ``nu_2_jt``, and
      ``psi_jt`` from ``(num_nodes, num_ticks)`` to
      ``(num_ticks, num_nodes)`` if the input came in
      location-major-axis form.
    - Handles ``epidemic_threshold`` as either a scalar or a
      length-``num_nodes`` array.
    - Reshapes ``beta_j0_env`` to ``(-1, 1)`` so downstream broadcast
      works correctly.
    - If ``epidemic_peaks`` is present, promotes it to a pandas
      DataFrame and appends a ``loc_idx`` column mapping each
      ``iso_code`` row to its 0-based index in
      ``params.location_name``.

    Note:
        The location order in the returned ``PropertySetEx`` is the
        order the locations appear in the input dict. The TODO at the
        top of the function flags this as a candidate for canonicalisation
        (alphabetical by name, or by integer ID) — not done yet.

    Args:
        parameters: A raw parameters dict, typically from
            ``json.load(default_parameters.json)`` or a user-supplied
            override file. Keys whose values would be coerced are stored
            on the result as attributes of the appropriate dtype; keys
            not in the coercion list (extras, future-compatibility
            fields) pass through unchanged.

    Returns:
        A [`PropertySetEx`][laser.cholera.metapop.params.PropertySetEx]
        with every recognised field typed for the simulation loop. Pass
        this to [`Model`][laser.cholera.metapop.model.Model] or run it
        through
        [`validate_parameters`][laser.cholera.metapop.params.validate_parameters]
        for invariant checks.

    Raises:
        AssertionError: If ``p`` is not integral, ``reported_cases`` /
            ``reported_deaths`` is neither a list nor an ndarray,
            ``epidemic_threshold`` is neither scalar nor list/ndarray,
            ``tau_i`` values fall outside ``[0, 1]``, or any
            ``epidemic_peaks.iso_code`` value is absent from
            ``location_name``.
    """
    # Note the following canonicalizes the order of the locations based on the
    # order in the JSON file.
    # We might consider either
    # a) alphabetical order by location name or
    # b) order by int(ID)

    params = PropertySetEx(parameters)

    # No processing of "params.seed"

    params.date_start = datetime.strptime(params.date_start, "%Y-%m-%d") if isinstance(params.date_start, (str,)) else params.date_start  # noqa: DTZ007
    params.date_stop = datetime.strptime(params.date_stop, "%Y-%m-%d") if isinstance(params.date_stop, (str,)) else params.date_stop  # noqa: DTZ007
    params.nticks = (params.date_stop - params.date_start).days + 1
    logger.info(f"Simulation calendar dates: {params.date_start} to {params.date_stop} ({params.nticks} ticks)")

    # Handle single location instance (incoming data is scalar rather than list)
    if not isinstance(params.location_name, list):
        params.location_name = [params.location_name]

    num_ticks = params.nticks
    num_nodes = len(params.location_name)

    assert int(params.p) == params.p, f"p must be an integer, but got {params.p}"

    assert isinstance(params.reported_cases, (list, np.ndarray)), (
        f"reported_cases must be a list of lists or a NumPy array, got {type(params.reported_cases)}"
    )
    assert isinstance(params.reported_deaths, (list, np.ndarray)), (
        f"reported_deaths must be a list of lists or a NumPy array, got {type(params.reported_deaths)}"
    )

    # scalars
    scalars = [
        ("phi_1", np.float32),
        ("phi_2", np.float32),
        ("omega_1", np.float32),
        ("omega_2", np.float32),
        ("iota", np.float32),
        ("gamma_1", np.float32),
        ("gamma_2", np.float32),
        ("epsilon", np.float32),
        ("rho", np.float32),
        ("sigma", np.float32),
        ("chi_endemic", np.float32),
        ("chi_epidemic", np.float32),
        # ("epidemic_threshold", np.float32),
        ("mobility_omega", np.float32),
        ("mobility_gamma", np.float32),
        ("p", np.int32),
        ("alpha_1", np.float32),
        ("alpha_2", np.float32),
        ("zeta_1", np.float32),
        ("zeta_2", np.float32),
        ("kappa", np.float32),
        ("decay_days_short", np.float32),
        ("decay_days_long", np.float32),
        ("decay_shape_1", np.float32),
        ("decay_shape_2", np.float32),
        ("delta_reporting_cases", np.int32),
        ("delta_reporting_deaths", np.int32),
        ("rho_deaths", np.float32),
    ]
    for name, transform in scalars:
        setattr(params, name, transform(getattr(params, name)))

    # arrays
    arrays = [
        ("N_j_initial", as_ndarray, np.uint32),
        ("S_j_initial", as_ndarray, np.uint32),
        ("E_j_initial", as_ndarray, np.uint32),
        ("I_j_initial", as_ndarray, np.uint32),
        ("R_j_initial", as_ndarray, np.uint32),
        ("V1_j_initial", as_ndarray, np.uint32),
        ("V2_j_initial", as_ndarray, np.uint32),
        ("b_jt", np.array, np.float32),
        ("d_jt", np.array, np.float32),
        ("nu_1_jt", np.array, np.float32),
        ("nu_2_jt", np.array, np.float32),
        # ("epidemic_threshold", np.array, np.float32),
        ("longitude", as_ndarray, np.float32),
        ("latitude", as_ndarray, np.float32),
        ("tau_i", as_ndarray, np.float32),
        ("beta_j0_hum", as_ndarray, np.float32),
        ("a_1_j", as_ndarray, np.float32),
        ("a_2_j", as_ndarray, np.float32),
        ("b_1_j", as_ndarray, np.float32),
        ("b_2_j", as_ndarray, np.float32),
        ("beta_j0_env", as_ndarray, np.float32),
        ("theta_j", as_ndarray, np.float32),
        ("psi_jt", np.array, np.float32),
        ("psi_star_a", np.array, np.float32),
        ("psi_star_b", np.array, np.float32),
        ("psi_star_z", np.array, np.float32),
        ("psi_star_k", np.array, np.float32),
        ("reported_cases", handle_nan, np.float32),
        ("reported_deaths", handle_nan, np.float32),
        ("beta_j0_tot", np.array, np.float32),
        ("p_beta", np.array, np.float32),
        ("prop_S_initial", np.array, np.float32),
        ("prop_E_initial", np.array, np.float32),
        ("prop_I_initial", np.array, np.float32),
        ("prop_R_initial", np.array, np.float32),
        ("prop_V1_initial", np.array, np.float32),
        ("prop_V2_initial", np.array, np.float32),
        ("mu_j_baseline", np.array, np.float32),
        ("mu_j_slope", np.array, np.float32),
        ("mu_j_epidemic_factor", np.array, np.float32),
        ("mu_jt", np.array, np.float32),
    ]
    for name, transform, dtype in arrays:
        setattr(params, name, transform(getattr(params, name), dtype=dtype))

    if params.b_jt.shape == (num_nodes, num_ticks):
        params.b_jt = np.array(params.b_jt.T)  # index on time, then location

    if params.d_jt.shape == (num_nodes, num_ticks):
        params.d_jt = np.array(params.d_jt.T)  # index on time, then location

    if params.nu_1_jt.shape == (num_nodes, num_ticks):
        params.nu_1_jt = np.array(params.nu_1_jt.T)  # index on time, then location

    if params.nu_2_jt.shape == (num_nodes, num_ticks):
        params.nu_2_jt = np.array(params.nu_2_jt.T)  # index on time, then location

    # epidemic_threshold: scalar or 1-D array of length num_nodes
    if isinstance(params.epidemic_threshold, Number):
        params.epidemic_threshold = np.float32(params.epidemic_threshold)
    else:
        assert isinstance(params.epidemic_threshold, (list, np.ndarray)), (
            f"epidemic_threshold must be a scalar or list of values, got {type(params.epidemic_threshold)}"
        )
        params.epidemic_threshold = np.asarray(params.epidemic_threshold, dtype=np.float32)

    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    # TODO - is this necessary?
    params.beta_j0_env = params.beta_j0_env.reshape(-1, 1)

    if params.psi_jt.shape == (num_nodes, num_ticks):
        params.psi_jt = np.array(params.psi_jt.T)  # index on time, then location

    if "epidemic_peaks" in params:
        params.epidemic_peaks = pd.DataFrame(params.epidemic_peaks)
        assert all(iso_code in params.location_name for iso_code in params.epidemic_peaks.iso_code)
        params.epidemic_peaks["loc_idx"] = [params.location_name.index(iso_code) for iso_code in params.epidemic_peaks.iso_code]

    return params


def validate_parameters(params: PropertySetEx) -> None:
    """Verify that a typed parameter set satisfies the model's invariants.

    Run as part of
    [`get_parameters`][laser.cholera.metapop.params.get_parameters] (when
    ``do_validation=True``, the default). The checks cover everything the
    simulation loop will subsequently assume but does not re-verify:
    consistent calendar dates, the right number of locations for every
    per-location vector and every per-tick-per-location matrix,
    non-negative compartment populations, rate/probability values
    confined to ``[0, 1]`` where required, the seasonality and mobility
    coefficient lists, the disease- and non-disease-mortality scaling
    arrays, and the ``epidemic_peaks`` DataFrame columns (when present).

    The function returns silently when every check passes; any failure
    is raised as ``AssertionError`` with a descriptive message naming
    the failed invariant.

    Args:
        params: A typed [`PropertySetEx`][laser.cholera.metapop.params.PropertySetEx]
            — typically the result of
            [`dict_to_propertysetex`][laser.cholera.metapop.params.dict_to_propertysetex]
            via
            [`get_parameters`][laser.cholera.metapop.params.get_parameters].

    Returns:
        None. Side effect on success is nothing; on failure, the
        offending invariant is raised.

    Raises:
        AssertionError: When any of the per-field invariants fails.
            Common categories: shape mismatch between a per-location
            vector and ``len(location_name)``; shape mismatch between a
            per-tick matrix and ``(nticks, npatches)``; out-of-range
            scalar (``phi_1`` / ``phi_2`` / ``rho`` / ``rho_deaths`` /
            ``sigma`` / ``alpha_1`` / ``alpha_2`` / ``theta_j`` /
            ``tau_i`` outside ``[0, 1]``); negative compartment
            population; missing required scalar
            (``mobility_omega`` / ``mobility_gamma`` / ``p``);
            ``decay_days_short > decay_days_long``;
            ``epidemic_peaks`` DataFrame missing the ``iso_code`` or
            ``peak_date`` column.
        RuntimeError: If ``epidemic_threshold`` is neither a scalar
            nor an ndarray (i.e., ingestion produced an unexpected
            type).
    """
    # date_start and date_stop
    assert params.date_stop >= params.date_start, f"date_stop ({params.date_stop}) must be >= date_start ({params.date_start})"

    npatches = len(params.location_name)

    assert params.S_j_initial.shape == (npatches,), (
        f"Number of S_j_initial values ({len(params.S_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.S_j_initial >= 0), "S_j_initial values must be non-negative"
    assert params.E_j_initial.shape == (npatches,), (
        f"Number of E_j_initial values ({len(params.E_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.E_j_initial >= 0), "E_j_initial values must be non-negative"
    assert params.I_j_initial.shape == (npatches,), (
        f"Number of I_j_initial values ({len(params.I_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.I_j_initial >= 0), "I_j_initial values must be non-negative"
    assert params.R_j_initial.shape == (npatches,), (
        f"Number of R_j_initial values ({len(params.R_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.R_j_initial >= 0), "R_j_initial values must be non-negative"
    assert params.V1_j_initial.shape == (npatches,), (
        f"Number of V1_j_initial values ({len(params.V1_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.V1_j_initial >= 0), "V1_j_initial values must be non-negative"
    assert params.V2_j_initial.shape == (npatches,), (
        f"Number of V2_j_initial values ({len(params.V2_j_initial)}) does not match number of locations ({npatches})"
    )
    assert np.all(params.V2_j_initial >= 0), "V2_j_initial values must be non-negative"

    nticks = params.nticks

    # shape of b_jt = (nticks, npatches)
    assert params.b_jt.shape == (nticks, npatches), f"Shape of b_jt {params.b_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    # 0 <= b_jt
    assert np.all(params.b_jt >= 0.0), "b_jt rate values must be positive"

    # shape of b_jt = (nticks, npatches)
    assert params.d_jt.shape == (nticks, npatches), f"Shape of d_jt {params.d_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    # 0 <= d_jt
    assert np.all(params.d_jt >= 0.0), "d_jt rate values must be positive"

    # shape of nu_1_jt = (nticks, npatches)
    assert params.nu_1_jt.shape == (nticks, npatches), (
        f"Shape of nu_1_jt {params.nu_1_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # # nu_1_jt - no daily value can be larger than the country population (N_j_initial) / 7
    # assert np.all(params.nu_1_jt <= params.N_j_initial[np.newaxis, :] / 7), (
    #     "nu_1_jt values must not exceed N_j_initial / 7 for any location"
    # )

    # shape of nu_2_jt = (nticks, npatches)
    assert params.nu_2_jt.shape == (nticks, npatches), (
        f"Shape of nu_2_jt {params.nu_2_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )
    # # nu_2_jt - no daily value can be larger than the country population (N_j_initial) / 7
    # assert np.all(params.nu_2_jt <= params.N_j_initial[np.newaxis, :] / 7), (
    #     "nu_2_jt values must not exceed N_j_initial / 7 for any location"
    # )

    # phi_1 and phi_2 must be between 0 (completely ineffective) and 1 (fully effective)
    assert (params.phi_1 >= 0.0) & (params.phi_1 <= 1.0), "phi_1 value must be in the range [0, 1]"
    assert (params.phi_2 >= 0.0) & (params.phi_2 <= 1.0), "phi_2 value must be in the range [0, 1]"

    # omega_1 and omega_2 must be above zero
    assert params.omega_1 >= 0.0, "omega_1 value must be positive"
    assert params.omega_2 >= 0.0, "omega_2 value must be positive"

    # iota must be above zero
    assert params.iota >= 0.0, "iota value must be positive"

    # gamma_1 and gamma_2 must be positive
    assert params.gamma_1 >= 0.0, "gamma_1 value must be positive"
    assert params.gamma_2 >= 0.0, "gamma_2 value must be positive"

    # epsilon must be positive
    assert params.epsilon >= 0.0, "epsilon value must be positive"

    if isinstance(params.epidemic_threshold, (Number, np.number)):
        assert params.epidemic_threshold >= 0, f"epidemic_threshold {params.epidemic_threshold} must be >= 0"
    elif isinstance(params.epidemic_threshold, np.ndarray):
        assert np.all(params.epidemic_threshold >= 0), f"epidemic_threshold values must be >= 0 ({params.epidemic_threshold.min()=})"
    else:
        raise RuntimeError(f"params.epidemic_threshold is of an unexpected type: {type(params.epidemic_threshold)}")

    assert params.mu_j_baseline.shape == (npatches,), f"Shape of params.mu_j_baseline ({params.mu_j_baseline.shape}) does not match ({npatches},)"
    assert np.all(params.mu_j_baseline >= 0), f"mu_j_baseline values must be >= 0 {params.mu_j_baseline.min()=}"

    assert params.mu_j_slope.shape == (npatches,), f"Shape of params.mu_j_slope ({params.mu_j_slope.shape}) does not match ({npatches},)"
    # no range constraints on mu_j_slope

    assert params.mu_j_epidemic_factor.shape == (npatches,), (
        f"Shape of params.mu_j_epidemic_factor ({params.mu_j_epidemic_factor.shape}) does not match ({npatches},)"
    )
    assert np.all(params.mu_j_epidemic_factor >= 0), f"mu_j_epidemic_factor values must be >= 0 ({params.mu_j_epidemic_factor.min()=})"

    assert params.delta_reporting_cases >= 0, f"delta_reporting_cases {params.delta_reporting_cases} must be >= 0"
    assert params.delta_reporting_deaths >= 0, f"delta_reporting_deaths {params.delta_reporting_deaths} must be >= 0"

    # rho must be between 0 (all false positives) and 1 (no false positives)
    assert (params.rho >= 0.0) & (params.rho <= 1.0), "rho value must be in the range [0, 1]"
    assert (params.rho_deaths >= 0.0) & (params.rho_deaths <= 1.0), "rho_deaths value must be in the range [0, 1]"

    # sigma must be between 0 (all asymptomatic) and 1 (all symptomatic)
    assert (params.sigma >= 0.0) & (params.sigma <= 1.0), "sigma value must be in the range [0, 1]"

    # Number of lat/long values must match number of patches
    assert len(params.latitude) == npatches, f"Number of latitude values ({len(params.latitude)}) does not match number of locations ({npatches})"
    assert len(params.longitude) == npatches, f"Number of longitude values ({len(params.longitude)}) does not match number of locations ({npatches})"
    # omega and gamma required to build pi_ij matrix with "power_norm"
    assert "mobility_omega" in params, "Parameters: 'mobility_omega' not found in parameters"
    assert "mobility_gamma" in params, "Parameters: 'mobility_gamma' not found in parameters"

    # Number of seasonality parameters must match number of patches
    assert len(params.a_1_j) == npatches, f"Number of a_1_j values ({len(params.a_1_j)}) does not match number of locations ({npatches})"
    assert len(params.b_1_j) == npatches, f"Number of b_1_j values ({len(params.b_1_j)}) does not match number of locations ({npatches})"
    assert len(params.a_2_j) == npatches, f"Number of a_2_j values ({len(params.a_2_j)}) does not match number of locations ({npatches})"
    assert len(params.b_2_j) == npatches, f"Number of b_2_j values ({len(params.b_2_j)}) does not match number of locations ({npatches})"
    assert "p" in params, "Parameters: 'p' (seasonality phase) not found in parameters"

    # length of beta_j0_hum must be equal to number of locations
    assert len(params.beta_j0_hum) == npatches, (
        f"Number of beta_j0_hum values ({len(params.beta_j0_hum)}) does not match number of locations ({npatches})"
    )
    # beta_j0_hum must be >= 0
    assert np.all(params.beta_j0_hum >= 0.0), "beta_j0_hum values must be >= 0"

    # length of tau_i must be equal to number of locations
    assert len(params.tau_i) == npatches, f"Number of tau_i values ({len(params.tau_i)}) does not match number of locations ({npatches})"
    # tau_i must be between 0 (no emigration) and 1 (all emigration)
    assert np.all((params.tau_i >= 0.0) & (params.tau_i <= 1.0)), "tau_i values must be in the range [0, 1]"

    # alpha_1 and alpha_2
    # TODO - TBD

    # alpha_1 must be above 0 (zero population mixing) and below 1 (full mass action), cannot equal zero
    assert (params.alpha_1 > 0.0) & (params.alpha_1 <= 1.0), "alpha_1 value must be in the range [0, 1]"

    # alpha_2 must be between 0 (full density dependence) and 1 (full frequency dependence)
    assert (params.alpha_2 >= 0.0) & (params.alpha_2 <= 1.0), "alpha_1 value must be in the range [0, 1]"

    # length of beta_j0_env must be equal to number of locations
    assert len(params.beta_j0_env) == npatches, (
        f"Number of beta_j0_env values ({len(params.beta_j0_env)}) does not match number of locations ({npatches})"
    )
    # beta_j0_env must be >= 0
    assert np.all(params.beta_j0_env >= 0.0), "beta_j0_env values must be >= 0"

    # length of theta_j must be equal to number of locations
    assert len(params.theta_j) == npatches, f"Number of theta_j values ({len(params.theta_j)}) does not match number of locations ({npatches})"
    # theta_j must be between 0 (no WASH intervention) and 1 (full WASH protection)
    assert np.all((params.theta_j >= 0.0) & (params.theta_j <= 1.0)), "theta_j values must be in the range [0, 1]"

    # shape of psi_jt = (nticks, npatches)
    assert params.psi_jt.shape == (nticks, npatches), (
        f"Shape of psi_jt {params.psi_jt.shape} does not match (nticks, npatches) = ({nticks}, {npatches})"
    )

    # psi_jt
    # TODO - TBD

    # zeta_1 and zeta_2 must be >= 0
    assert params.zeta_1 >= 0.0, "zeta_1 value must be >= 0"
    assert params.zeta_2 >= 0.0, "zeta_2 value must be >= 0"
    # TODO - TBD any other limits

    # kappa must be >= 0
    assert params.kappa >= 0.0, "kappa value must be >= 0"

    # decay_days_short > 0.0
    assert params.decay_days_short > 0.0, f"decay_days_short value must be > 0 {params.decay_days_short=}"
    # decay_days_short <= decay_days_long
    assert params.decay_days_short <= params.decay_days_long, (
        f"decay_days_short ({params.decay_days_short}) value must be <= decay_days_long ({params.decay_days_long})"
    )

    if "epidemic_peaks" in params:
        assert isinstance(params.epidemic_peaks, pd.DataFrame), (
            f"'epidemic_peaks' should be convertable to a Pandas DataFrame, found {type(params.epidemic_peaks)}"
        )
        assert "iso_code" in params.epidemic_peaks.columns, f"'epidemic_peaks' should contain 'iso_code' column, {params.epidemic_peaks.columns=}"
        assert "peak_date" in params.epidemic_peaks.columns, f"'epidemic_peaks' should contain 'peak_date' column, {params.epidemic_peaks.columns=}"

    return


class Parameters:
    """Pipeline component whose only job is to render an extensive parameter overview.

    Has no per-tick work — `__call__` is a no-op. The
    [`plot`][laser.cholera.metapop.params.Parameters.plot] generator
    yields nine figures (initial populations, four rate heatmaps,
    emigration/WASH scatters, mortality and suitability heatmaps) that
    surface the input parameters for human review.

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
        """Validate that `model.params` is attached.

        Raises:
            AssertionError: When `model.params` is missing.
        """
        # assert hasattr(self.model, "patches"), "Parameters: model needs to have a 'patches' attribute."
        # assert hasattr(self.model, "people"), "Parameters: model needs to have a 'people' attribute."
        assert hasattr(self.model, "params"), "Parameters: model needs to have a 'params' attribute."

        return

    def __call__(self, _model, _tick):
        """No-op per-tick callable; the component only contributes to the post-run plot."""

    def plot(self, fig: Figure = None) -> Iterator[str]:  # pragma: no cover
        """Yield nine Matplotlib figures summarizing the input parameters.

        Generates (in order): initial-population stacked bar; `b_jt` /
        `d_jt` / `nu_1_jt` / `nu_2_jt` / `mu_jt` per-patch-per-tick
        heatmaps; `tau_i` and `theta_j` per-patch scatter; `psi_jt`
        suitability heatmap.

        Args:
            fig: Optional existing Matplotlib `Figure` to draw into.

        Yields:
            Nine labels in order: `"Initial Populations by Category"`,
            `"Birth Rates by Location Over Time"`,
            `"Non-Disease Mortality Rates by Location Over Time"`,
            `"First Dose Vaccination Counts by Location Over Time"`,
            `"Second Dose Vaccination Counts by Location Over Time"`,
            `"Disease Mortality Rate by Location Over Time"`,
            `"Emigration Probabilities by Location"`,
            `"WASH Coverage by Location"`,
            `"Environmental Suitability Factor by Location Over Time"`.
        """
        # Stacked bar chart of initial populations
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Initial Populations by Category") if fig is None else fig

        categories = ["S_j_initial", "E_j_initial", "I_j_initial", "R_j_initial", "V1_j_initial", "V2_j_initial"]
        data = [getattr(self.model.params, category) for category in categories]

        x = np.arange(len(self.model.params.location_name))
        bottom = np.zeros(len(self.model.params.location_name))

        for category, values in zip(categories, data, strict=True):
            plt.bar(x, values, bottom=bottom, label=category)
            bottom += values

        plt.xticks(x, self.model.params.location_name, rotation=45, ha="right")
        plt.xlabel("Location Name")
        plt.ylabel("Population")
        plt.legend()

        yield "Initial Populations by Category"

        # Birth rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Birth Rates by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.b_jt.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Birth Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Birth Rates by Location Over Time"

        # Mortality rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Non-Disease Mortality Rates by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.d_jt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Mortality Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Non-Disease Mortality Rates by Location Over Time"

        # Vaccination (first dose) rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="First Dose Vaccination Counts by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.nu_1_jt.T, aspect="auto", cmap="Greens", interpolation="nearest")
        plt.colorbar(label="Vaccination Count")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "First Dose Vaccination Counts by Location Over Time"

        # Vaccination (second dose) rates by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Second Dose Vaccination Counts by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.nu_2_jt.T, aspect="auto", cmap="Greens", interpolation="nearest")
        plt.colorbar(label="Vaccination Count")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Second Dose Vaccination Counts by Location Over Time"

        # Disease mortality rate over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Disease Mortality Rate by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.mu_jt.T, aspect="auto", cmap="Reds", interpolation="nearest")
        plt.colorbar(label="Disease Mortality Rate")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Disease Mortality Rate by Location Over Time"

        # Emmigration probability rates by location
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Emigration Probabilities by Location") if fig is None else fig

        plt.scatter(self.model.params.location_name, self.model.params.tau_i, marker="x", color="purple")
        plt.xlabel("Location Name")
        plt.ylabel("Emigration Probability")
        plt.xticks(rotation=45, ha="right")

        yield "Emigration Probabilities by Location"

        # WASH fraction by location
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="WASH Coverage by Location") if fig is None else fig

        plt.scatter(self.model.params.location_name, self.model.params.theta_j, marker="x", color="purple")
        plt.xlabel("Location Name")
        plt.ylabel("WASH Coverage")
        plt.xticks(rotation=45, ha="right")

        yield "WASH Coverage by Location"

        # Environmental suitability factor by location over time
        _fig = plt.figure(figsize=(12, 9), dpi=128, num="Environmental Suitability Factor by Location Over Time") if fig is None else fig

        plt.imshow(self.model.params.psi_jt.T, aspect="auto", cmap="Blues", interpolation="nearest")
        plt.colorbar(label="Environmental Suitability Factor")
        plt.xlabel("Time (Days)")
        plt.ylabel("Location")
        plt.yticks(ticks=np.arange(len(self.model.params.location_name)), labels=self.model.params.location_name)

        yield "Environmental Suitability Factor by Location Over Time"

        return
