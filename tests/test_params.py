"""Tests for laser.cholera.metapop.params: parameter ingestion and validation.

These tests cover loading the bundled default parameter files, the conversion of
``epidemic_peaks`` from raw dict-like input into a pandas DataFrame on ingestion,
and the column-presence checks enforced by ``validate_parameters``.
"""

import json
from pathlib import Path

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


class TestEpidemicPeaksIngestion:
    """Tests for ``epidemic_peaks`` ingestion in ``dict_to_propertysetex``.

    ``epidemic_peaks`` is optional but, when present, must be promoted to a
    pandas DataFrame with ``iso_code`` and ``peak_date`` columns so that the
    likelihood code can filter by ISO code. These tests pin that contract.
    """

    def test_list_of_dicts_input_becomes_dataframe(self):
        """A list-of-dicts ``epidemic_peaks`` (the raw JSON shape) becomes a DataFrame.

        Given default parameters whose ``epidemic_peaks`` field is a list of
        ``{iso_code, peak_date}`` dicts (the on-disk JSON format),
        when ``get_parameters`` ingests them,
        then ``params.epidemic_peaks`` should be a pandas DataFrame with the
        expected ``iso_code`` and ``peak_date`` columns and the row count
        should equal the number of input entries.

        Failure implies the conversion in ``dict_to_propertysetex`` did not run
        or did not preserve the columns the likelihood code expects.
        """
        raw = _load_default_dict()
        assert isinstance(raw["epidemic_peaks"], list), (
            "default_parameters.json should provide epidemic_peaks as a list of dicts; "
            "if this changed the test setup needs updating"
        )
        expected_rows = len(raw["epidemic_peaks"])

        params = get_parameters(DEFAULT_PARAMS_JSON, mods={"loglevel": "DEBUG"})

        assert isinstance(params.epidemic_peaks, pd.DataFrame)
        assert list(params.epidemic_peaks.columns) == ["iso_code", "peak_date"]
        assert len(params.epidemic_peaks) == expected_rows

    def test_dict_of_lists_input_becomes_dataframe(self):
        """A dict-of-lists ``epidemic_peaks`` is also promoted to a DataFrame.

        Given a parameter dict whose ``epidemic_peaks`` field is a
        ``{column_name: [values]}`` mapping (an alternative serialization shape
        that pandas accepts),
        when ``dict_to_propertysetex`` ingests it,
        then ``params.epidemic_peaks`` should be a DataFrame with the same two
        columns and the values from the input lists.

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
        assert set(params.epidemic_peaks.columns) == {"iso_code", "peak_date"}
        assert len(params.epidemic_peaks) == 3
        assert params.epidemic_peaks.iloc[0]["iso_code"] == "AGO"

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
