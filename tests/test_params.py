"""Tests for laser.cholera.metapop.params: parameter ingestion and validation.

These tests cover loading the bundled default parameter files, the conversion of
``epidemic_peaks`` from raw dict-like input into a pandas DataFrame on ingestion,
and the column-presence checks enforced by ``validate_parameters``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from laser.cholera.metapop.params import dict_to_propertysetex
from laser.cholera.metapop.params import get_parameters
from laser.cholera.metapop.params import validate_parameters

SCRIPT_DIR = Path(__file__).parent.absolute()
PARAMS_DIR = SCRIPT_DIR / "../src/laser/cholera/metapop/data"
DEFAULT_PARAMS_JSON = PARAMS_DIR / "default_parameters.json"


def _load_default_dict() -> dict:
    """Return a fresh dict copy of the bundled default parameters JSON."""
    with DEFAULT_PARAMS_JSON.open("r") as fp:
        return json.load(fp)


class TestGetParameters:
    """Tests for ``get_parameters`` loading the bundled JSON files.

    Failure of these tests implies the default parameter file format has drifted
    from what ``dict_to_propertysetex`` expects, or that validation is now
    rejecting the canonical bundled configuration.
    """

    def test_load_uncompressed_json(self):
        """Loading the uncompressed default JSON returns a usable parameter set.

        Given the bundled ``default_parameters.json`` file,
        when ``get_parameters`` is called with a debug logging mod,
        then it should complete without raising.

        Failure implies the default file no longer satisfies the validation rules
        (e.g., a required field was removed or its type changed).
        """
        _params = get_parameters(DEFAULT_PARAMS_JSON, mods={"loglevel": "DEBUG"})

    def test_load_compressed_json(self):
        """Loading the gzip-compressed default JSON returns a usable parameter set.

        Given the bundled ``default_parameters.json.gz`` file,
        when ``get_parameters`` is called with a debug logging mod,
        then it should complete without raising.

        Failure implies the compressed copy has drifted from the uncompressed one
        or the gzip ingestion path is broken.
        """
        _params = get_parameters(PARAMS_DIR / "default_parameters.json.gz", mods={"loglevel": "DEBUG"})

    @pytest.mark.parametrize(
        "suffix",
        [".h5", ".hdf", ".hdf5", ".h5.gz", ".hdf.gz", ".hdf5.gz"],
    )
    def test_hdf5_config_paths_no_longer_supported(self, suffix):
        """Config-parameter ingestion from HDF5 files is rejected.

        Given a ``get_parameters`` call with a file path whose suffix is one of
        the historical HDF5 variants (``.h5``, ``.hdf``, ``.hdf5``, and their
        ``.gz`` compressed forms),
        when the function attempts to dispatch on suffix,
        then it should raise a ``KeyError`` because the loader entry has been
        removed from ``fn_map``.

        Failure implies HDF5 config-parameter loading has been silently
        re-introduced. This test does not require the file to exist — the
        suffix-based dispatch error is raised before any disk access.

        Inconsistency note: the propagated error type is ``KeyError`` from a
        raw dict lookup rather than a domain-specific exception with a helpful
        message. If a cleaner error is added later, this test can be tightened
        to assert on the new exception type / message.
        """
        # Use a deliberately non-existent path so we don't accidentally read
        # any real HDF5 fixtures lying around in the data directory.
        bogus = Path("/nonexistent") / f"params{suffix}"

        with pytest.raises(KeyError):
            get_parameters(bogus)


class TestEpidemicPeaksIngestion:
    """Tests for ``epidemic_peaks`` ingestion in ``dict_to_propertysetex``.

    ``epidemic_peaks`` is optional but, when present, must be promoted to a
    pandas DataFrame with ``iso_code``, ``peak_date``, and ``loc_idx`` columns
    so that the likelihood code can dispatch each peak to the correct
    simulation row by integer index. These tests pin that contract.
    """

    def test_list_of_dicts_input_becomes_dataframe(self):
        """A list-of-dicts ``epidemic_peaks`` (the raw JSON shape) becomes a DataFrame.

        Given default parameters whose ``epidemic_peaks`` field is a list of
        ``{iso_code, peak_date}`` dicts (the on-disk JSON format),
        when ``get_parameters`` ingests them,
        then ``params.epidemic_peaks`` should be a pandas DataFrame whose
        columns are exactly ``iso_code``, ``peak_date``, ``loc_idx`` (in that
        order), and the row count should equal the number of input entries.

        Failure implies the conversion in ``dict_to_propertysetex`` did not run
        or did not produce the columns the likelihood code expects.
        """
        raw = _load_default_dict()
        assert isinstance(raw["epidemic_peaks"], list), (
            "default_parameters.json should provide epidemic_peaks as a list of dicts; if this changed the test setup needs updating"
        )
        expected_rows = len(raw["epidemic_peaks"])

        params = get_parameters(DEFAULT_PARAMS_JSON, mods={"loglevel": "DEBUG"})

        assert isinstance(params.epidemic_peaks, pd.DataFrame)
        assert list(params.epidemic_peaks.columns) == ["iso_code", "peak_date", "loc_idx"]
        assert len(params.epidemic_peaks) == expected_rows

    def test_dict_of_lists_input_becomes_dataframe(self):
        """A dict-of-lists ``epidemic_peaks`` is also promoted to a DataFrame.

        Given a parameter dict whose ``epidemic_peaks`` field is a
        ``{column_name: [values]}`` mapping (an alternative serialization shape
        that pandas accepts) with ISO codes that exist in ``location_name``,
        when ``dict_to_propertysetex`` ingests it,
        then ``params.epidemic_peaks`` should be a DataFrame with the input
        columns plus the computed ``loc_idx`` column, and the row count should
        match the input.

        Failure implies the conversion is too narrow and only accepts one
        serialization, which would break callers that pass the columnar form.
        """
        raw = _load_default_dict()
        raw["epidemic_peaks"] = {
            "iso_code": ["AGO", "BDI", "CMR"],
            "peak_date": ["2018-01-04", "2017-10-17", "2010-08-29"],
        }

        params = dict_to_propertysetex(raw)

        assert isinstance(params.epidemic_peaks, pd.DataFrame)
        assert set(params.epidemic_peaks.columns) == {"iso_code", "peak_date", "loc_idx"}
        assert len(params.epidemic_peaks) == 3
        assert params.epidemic_peaks.iloc[0]["iso_code"] == "AGO"

    def test_loc_idx_column_maps_iso_code_to_location_name_index(self):
        """The ingested ``loc_idx`` matches ``location_name.index(iso_code)`` for every row.

        Given default parameters whose ``epidemic_peaks`` contains ISO codes
        that all appear in ``location_name`` (the canonical bundled config),
        when ``get_parameters`` ingests them,
        then for every row in ``params.epidemic_peaks`` the ``loc_idx`` value
        must equal the position of its ``iso_code`` inside
        ``params.location_name``. The check uses a recomputed expected vector
        rather than spot-checking known indices so the test pins the *rule*,
        not the current default data.

        Failure implies the ``loc_idx`` column is missing, mis-aligned with
        the source iso codes, or computed against the wrong location list — any
        of which would silently route peaks to the wrong simulation row in
        ``calc_model_likelihood``.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, mods={"loglevel": "DEBUG"})

        assert "loc_idx" in params.epidemic_peaks.columns
        expected = [params.location_name.index(code) for code in params.epidemic_peaks["iso_code"]]
        actual = params.epidemic_peaks["loc_idx"].tolist()
        assert actual == expected

        # Additional spot check: ``loc_idx`` is an integer index in
        # ``[0, len(location_name))`` for every row.
        n_locations = len(params.location_name)
        assert all(0 <= int(idx) < n_locations for idx in actual)

    def test_unknown_iso_code_in_epidemic_peaks_fails_assert(self):
        """An ISO code in ``epidemic_peaks`` that is absent from ``location_name`` raises.

        Given default parameters whose ``epidemic_peaks`` has an extra row with
        an ISO code (``"ZZZ"``) that does not appear in ``location_name``,
        when ``dict_to_propertysetex`` ingests the dict,
        then the assertion in ``params.py`` that every ``iso_code`` is present
        in ``location_name`` must fail with an ``AssertionError``.

        Failure implies unknown ISO codes are silently passed through to the
        downstream ``loc_idx`` computation, where ``list.index`` would raise a
        ``ValueError`` later — or worse, in a future refactor, the unknown
        code could land on a wrong index and miscredit peaks to the wrong
        simulation row.
        """
        raw = _load_default_dict()
        # Inconsistency note: the existing assert message is implicit (just
        # ``assert all(...)``), so we only match on AssertionError rather than
        # a specific text. If a more descriptive message is added later, this
        # test will still pass.
        raw["epidemic_peaks"] = raw["epidemic_peaks"] + [{"iso_code": "ZZZ", "peak_date": "2024-01-01"}]
        assert "ZZZ" not in raw["location_name"], "Test setup assumes ZZZ is not a valid ISO code in default location_name"

        with pytest.raises(AssertionError):
            dict_to_propertysetex(raw)

    def test_missing_epidemic_peaks_is_allowed(self):
        """An absent ``epidemic_peaks`` entry is allowed and yields no attribute.

        Given default parameters with the ``epidemic_peaks`` key removed,
        when ``dict_to_propertysetex`` ingests them and ``validate_parameters``
        is then called,
        then the result should not have an ``epidemic_peaks`` attribute and
        validation should succeed.

        Failure implies ``epidemic_peaks`` was inadvertently made mandatory, or
        that the optional branch in validation is being triggered with no data.
        """
        raw = _load_default_dict()
        raw.pop("epidemic_peaks", None)

        params = dict_to_propertysetex(raw)
        validate_parameters(params)

        assert "epidemic_peaks" not in params


class TestEpidemicPeaksValidation:
    """Tests for the ``epidemic_peaks`` checks in ``validate_parameters``.

    The likelihood code filters peaks by ``iso_code`` and reads ``peak_date``,
    so validation must catch the case where either column is absent before the
    bad data reaches downstream consumers.
    """

    def test_validate_passes_for_well_formed_dataframe(self):
        """A well-formed ``epidemic_peaks`` DataFrame passes validation.

        Given the bundled default parameters (which include a valid
        ``epidemic_peaks`` DataFrame after ingestion),
        when ``validate_parameters`` is called,
        then it should return without raising.

        Failure implies a regression where the canonical default configuration
        is now flagged as invalid.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False, mods={"loglevel": "DEBUG"})
        # Inconsistency note: ``do_validation=False`` is used to isolate the
        # call we are exercising; ingestion still converts ``epidemic_peaks``
        # to a DataFrame.
        assert isinstance(params.epidemic_peaks, pd.DataFrame)

        validate_parameters(params)

    def test_validate_fails_when_iso_code_column_missing(self):
        """Missing ``iso_code`` column triggers an AssertionError.

        Given default parameters whose ``epidemic_peaks`` DataFrame has had its
        ``iso_code`` column renamed away (simulating an upstream schema error),
        when ``validate_parameters`` is called,
        then it should raise an AssertionError mentioning ``iso_code``.

        Failure implies validation no longer catches the missing-column case,
        which would let bad data through to the likelihood code where it would
        produce a less obvious KeyError or silent zero result.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False, mods={"loglevel": "DEBUG"})
        params.epidemic_peaks = params.epidemic_peaks.rename(columns={"iso_code": "country"})

        with pytest.raises(AssertionError, match="iso_code"):
            validate_parameters(params)

    def test_validate_fails_when_peak_date_column_missing(self):
        """Missing ``peak_date`` column triggers an AssertionError.

        Given default parameters whose ``epidemic_peaks`` DataFrame has had its
        ``peak_date`` column renamed away,
        when ``validate_parameters`` is called,
        then it should raise an AssertionError mentioning ``peak_date``.

        Failure implies validation no longer catches the missing-column case,
        with the same consequences as ``iso_code`` going missing.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False, mods={"loglevel": "DEBUG"})
        params.epidemic_peaks = params.epidemic_peaks.rename(columns={"peak_date": "date"})

        with pytest.raises(AssertionError, match="peak_date"):
            validate_parameters(params)


class TestAlphaDualMode:
    """Tests for ``alpha_1`` / ``alpha_2`` accepting either a scalar or a length-``npatches`` array.

    Both parameters drive `np.power(..., alpha_*)` in ``humantohuman.py``; the
    consumer broadcasts either form, so the ingestion path must accept both and
    the validator must enforce shape + range invariants on both.
    """

    def test_alpha_scalar_round_trip(self):
        """A scalar ``alpha_1`` / ``alpha_2`` in the input dict survives as a ``np.float32`` scalar.

        Given the bundled defaults (which ship with scalar alphas),
        when ``get_parameters`` ingests them,
        then ``params.alpha_1`` and ``params.alpha_2`` are ``np.float32`` scalars,
        validation passes, and the values round-trip from the input JSON.

        Failure implies the dual-mode ingestion branch regressed and is no longer
        accepting the existing scalar-only configurations.
        """
        raw = _load_default_dict()
        raw["alpha_1"] = 0.85
        raw["alpha_2"] = 0.95

        params = dict_to_propertysetex(raw)
        validate_parameters(params)

        assert isinstance(params.alpha_1, np.floating)
        assert isinstance(params.alpha_2, np.floating)
        assert float(params.alpha_1) == pytest.approx(0.85)
        assert float(params.alpha_2) == pytest.approx(0.95)

    def test_alpha_array_round_trip(self):
        """A length-``npatches`` ``alpha_1`` / ``alpha_2`` array is coerced to ``np.ndarray[np.float32]``.

        Given default parameters where ``alpha_1`` / ``alpha_2`` are replaced
        with length-``npatches`` lists of in-range per-patch values,
        when ``get_parameters`` ingests them,
        then both end up as ``np.ndarray`` with shape ``(npatches,)`` and dtype
        ``float32``, and validation passes.

        Failure implies the dual-mode ingestion branch regressed for the per-patch
        scenario — the consumer in ``humantohuman.__call__`` would then receive
        the wrong shape and either crash or broadcast incorrectly.
        """
        raw = _load_default_dict()
        npatches = len(raw["location_name"])
        raw["alpha_1"] = [0.85] * npatches
        raw["alpha_2"] = [0.95] * npatches

        params = dict_to_propertysetex(raw)
        validate_parameters(params)

        assert isinstance(params.alpha_1, np.ndarray)
        assert params.alpha_1.shape == (npatches,)
        assert params.alpha_1.dtype == np.float32
        assert isinstance(params.alpha_2, np.ndarray)
        assert params.alpha_2.shape == (npatches,)
        assert params.alpha_2.dtype == np.float32

    @pytest.mark.parametrize("name", ["alpha_1", "alpha_2"])
    def test_alpha_wrong_length_array_rejected_at_ingestion(self, name):
        """A wrong-length ``alpha_1`` / ``alpha_2`` array is rejected during ingestion.

        Given default parameters with ``alpha_*`` replaced by a list whose
        length is ``npatches + 1`` (a common upstream-data mistake),
        when ``dict_to_propertysetex`` runs,
        then it raises an ``AssertionError`` naming the shape mismatch.

        Failure implies the per-patch shape guard regressed and a wrong-length
        array would silently flow into the consumer where ``np.power`` would
        either broadcast-error at runtime or — worse — produce a result of an
        unexpected shape.
        """
        raw = _load_default_dict()
        npatches = len(raw["location_name"])
        raw[name] = [0.5] * (npatches + 1)

        with pytest.raises(AssertionError, match=f"{name} array shape"):
            dict_to_propertysetex(raw)

    def test_alpha_1_scalar_zero_rejected_by_validator(self):
        """A scalar ``alpha_1 == 0`` is rejected (strict ``>`` lower bound).

        Given the default parameters with ``alpha_1`` set to ``0.0``,
        when ``validate_parameters`` runs,
        then it raises an ``AssertionError`` naming the ``(0, 1]`` range.

        Failure implies the lower-bound guard relaxed from strict ``>`` to ``>=``;
        ``alpha_1 = 0`` collapses ``np.power(effective_i, 0)`` to ``1`` and
        breaks the FOI dependence on infected counts.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False)
        params.alpha_1 = np.float32(0.0)

        with pytest.raises(AssertionError, match=r"alpha_1 scalar .* must be in \(0, 1\]"):
            validate_parameters(params)

    def test_alpha_1_array_with_zero_entry_rejected(self):
        """An ``alpha_1`` array containing a zero entry is rejected.

        Given default parameters with an in-range ``alpha_1`` array except for
        a single zero entry,
        when ``validate_parameters`` runs,
        then it raises an ``AssertionError`` referencing the ``(0, 1]`` range.

        Failure implies the per-element range guard regressed and a zero entry
        would silently pass through, breaking the FOI for that one patch.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False)
        npatches = len(params.location_name)
        params.alpha_1 = np.full(npatches, 0.85, dtype=np.float32)
        params.alpha_1[0] = 0.0

        with pytest.raises(AssertionError, match=r"alpha_1 array values must be in \(0, 1\]"):
            validate_parameters(params)

    @pytest.mark.parametrize("bad_value", [-0.1, 1.1])
    def test_alpha_2_scalar_out_of_range_rejected(self, bad_value):
        """A scalar ``alpha_2`` outside ``[0, 1]`` is rejected.

        Given the default parameters with ``alpha_2`` set to a value below 0
        or above 1,
        when ``validate_parameters`` runs,
        then it raises an ``AssertionError`` naming the ``[0, 1]`` range.

        Failure implies the range guard regressed; an out-of-range ``alpha_2``
        would skew the population-scaling exponent into a regime the model
        was not designed for.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False)
        params.alpha_2 = np.float32(bad_value)

        with pytest.raises(AssertionError, match=r"alpha_2 scalar .* must be in \[0, 1\]"):
            validate_parameters(params)

    @pytest.mark.parametrize("name", ["alpha_1", "alpha_2"])
    def test_alpha_wrong_length_array_rejected_at_validation(self, name):
        """A wrong-length ``alpha_*`` array set directly on the params is rejected by the validator.

        Given a parameter set whose ``alpha_1`` / ``alpha_2`` is replaced after
        ingestion with a wrong-length ``np.ndarray`` (bypassing the ingestion
        shape guard),
        when ``validate_parameters`` runs,
        then it raises an ``AssertionError`` naming the shape mismatch.

        This is the belt-and-braces check that catches direct in-Python mutations
        like ``params.alpha_1 = np.zeros(npatches - 1)`` which never go through
        ``dict_to_propertysetex``. Parallel to the equivalent check on
        ``epidemic_threshold``.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False)
        npatches = len(params.location_name)
        # Use 0.5 — in range for both alpha_1 (0, 1] and alpha_2 [0, 1] — so
        # the failure mode is purely shape, not range.
        setattr(params, name, np.full(npatches - 1, 0.5, dtype=np.float32))

        with pytest.raises(AssertionError, match=f"{name} array shape"):
            validate_parameters(params)

    def test_alpha_2_array_out_of_range_rejected(self):
        """An ``alpha_2`` array with any out-of-range entry is rejected.

        Given default parameters with an otherwise-valid ``alpha_2`` array
        except for one entry above 1,
        when ``validate_parameters`` runs,
        then it raises an ``AssertionError`` referencing the ``[0, 1]`` range
        and the offending min / max values.

        Failure implies the per-element range guard regressed.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False)
        npatches = len(params.location_name)
        params.alpha_2 = np.full(npatches, 0.5, dtype=np.float32)
        params.alpha_2[0] = 1.5

        with pytest.raises(AssertionError, match=r"alpha_2 array values must be in \[0, 1\]"):
            validate_parameters(params)


class TestEpidemicThresholdShape:
    """Tests for the tightened ``epidemic_threshold`` shape guard.

    ``epidemic_threshold`` accepts either a scalar or a length-``npatches``
    array (consumed in ``infectious.py`` against the per-patch infected
    fraction). Before this change, the array form was not length-checked at
    ingestion — a wrong-length array would either crash later inside
    ``np.where`` or, worse, silently broadcast into a wrong-shape result.
    """

    def test_epidemic_threshold_scalar_round_trip(self):
        """A scalar ``epidemic_threshold`` survives as a ``np.float32`` scalar.

        Given the bundled defaults (which ship with a scalar ``epidemic_threshold``),
        when ``get_parameters`` ingests them,
        then ``params.epidemic_threshold`` is a ``np.float32`` scalar and
        validation passes.

        Failure implies the dual-mode ingestion branch regressed for the
        scalar case.
        """
        raw = _load_default_dict()
        raw["epidemic_threshold"] = 0.05

        params = dict_to_propertysetex(raw)
        validate_parameters(params)

        assert isinstance(params.epidemic_threshold, np.floating)
        assert float(params.epidemic_threshold) == pytest.approx(0.05)

    def test_epidemic_threshold_array_round_trip(self):
        """A length-``npatches`` ``epidemic_threshold`` array is coerced to ``np.ndarray[np.float32]``.

        Given default parameters with ``epidemic_threshold`` replaced by a
        length-``npatches`` list of non-negative per-patch thresholds,
        when ``get_parameters`` ingests them,
        then ``params.epidemic_threshold`` is a ``np.ndarray`` of shape
        ``(npatches,)`` and dtype ``float32``, and validation passes.

        Failure implies the per-patch ingestion branch regressed.
        """
        raw = _load_default_dict()
        npatches = len(raw["location_name"])
        raw["epidemic_threshold"] = [0.05] * npatches

        params = dict_to_propertysetex(raw)
        validate_parameters(params)

        assert isinstance(params.epidemic_threshold, np.ndarray)
        assert params.epidemic_threshold.shape == (npatches,)
        assert params.epidemic_threshold.dtype == np.float32

    def test_epidemic_threshold_wrong_length_array_rejected_at_ingestion(self):
        """A wrong-length ``epidemic_threshold`` array is rejected during ingestion.

        Given default parameters with ``epidemic_threshold`` replaced by a list
        whose length is ``npatches - 1``,
        when ``dict_to_propertysetex`` runs,
        then it raises an ``AssertionError`` naming the shape mismatch.

        Failure implies the shape guard regressed and a wrong-length array
        would silently flow into ``infectious.py``'s ``np.where`` comparison,
        producing either a broadcast error or a silently mis-aligned regime
        mask.
        """
        raw = _load_default_dict()
        npatches = len(raw["location_name"])
        raw["epidemic_threshold"] = [0.05] * (npatches - 1)

        with pytest.raises(AssertionError, match="epidemic_threshold array shape"):
            dict_to_propertysetex(raw)

    def test_epidemic_threshold_wrong_length_array_rejected_at_validation(self):
        """A wrong-length ``epidemic_threshold`` array set directly on the params is rejected by the validator.

        Given a parameter set whose ``epidemic_threshold`` is replaced after
        ingestion with a wrong-length ``np.ndarray`` (bypassing the ingestion
        shape guard),
        when ``validate_parameters`` runs,
        then it raises an ``AssertionError`` naming the shape mismatch.

        This is the belt-and-braces check that catches direct in-Python
        mutations like ``params.epidemic_threshold = np.zeros(npatches - 1)``
        which never go through ``dict_to_propertysetex``.
        """
        params = get_parameters(DEFAULT_PARAMS_JSON, do_validation=False)
        npatches = len(params.location_name)
        params.epidemic_threshold = np.zeros(npatches - 1, dtype=np.float32)

        with pytest.raises(AssertionError, match="epidemic_threshold array shape"):
            validate_parameters(params)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
