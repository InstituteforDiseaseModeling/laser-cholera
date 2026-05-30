"""Tests for laser.cholera.metapop.recorder.

The Recorder writes the final tick's state of ``model.people`` and ``model.patches``
to an HDF5 file when (a) the current tick is the final tick of the simulation,
(b) ``params.hdf5_output`` is truthy, and (c) ``params["return"]`` is a truthy
collection of property names. These tests exercise the gate, the filename and
output-directory rules, the optional gzip compression, the selective property
serialization, and the error path when the model is missing ``people`` or
``patches``.
"""

import gzip
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from laser.cholera.metapop.params import PropertySetEx
from laser.cholera.metapop.recorder import Recorder
from laser.cholera.metapop.recorder import save_compressed_hdf5_parameters
from laser.cholera.metapop.recorder import save_hdf5_parameters


class StubFrame:
    """Stand-in for ``model.people`` / ``model.patches`` exposing a realistic
    mix of public arrays, an underscore-prefixed array, and a bound method so
    the recorder's filters have something to do.
    """

    def __init__(self, susceptible: np.ndarray, infected: np.ndarray) -> None:
        self.susceptible = susceptible
        self.infected = infected
        self._private = np.array([0], dtype=np.int32)  # excluded by underscore filter

    def helper_method(self) -> None:  # excluded by MethodType filter
        return None


def _make_params(**kwargs) -> PropertySetEx:
    """Build a minimal PropertySetEx with the keys the recorder consults.

    ``nticks`` defaults to 5 so the "final tick" is 4 in most tests. Every
    other key is opt-in so each test can dial in exactly the gate it wants
    to exercise.
    """
    payload = {"nticks": 5}
    payload.update(kwargs)
    return PropertySetEx(payload)


def _make_model(params: PropertySetEx, people=None, patches=None) -> SimpleNamespace:
    """Compose a stub model object exposing the attributes the recorder reads."""
    if people is None:
        people = StubFrame(np.array([100, 99, 98], dtype=np.int32), np.array([1, 2, 3], dtype=np.int32))
    if patches is None:
        patches = StubFrame(np.array([1000, 1001], dtype=np.int32), np.array([10, 11], dtype=np.int32))
    return SimpleNamespace(params=params, people=people, patches=patches)


class TestRecorderInitAndCheck:
    """Construction and ``check`` behaviour.

    Failure of these tests implies either ``__init__`` is not storing the model
    reference (so downstream methods would NameError) or ``check`` has gained a
    raise-on-missing behavior that it does not currently have.
    """

    def test_init_stores_model_reference(self):
        """``Recorder(model).model`` is the same object that was passed in.

        Given an arbitrary model stub,
        when a Recorder is constructed,
        then the recorder's ``.model`` attribute is the exact same object
        (identity, not equality).

        Failure implies the constructor has started copying or wrapping the
        model, which would break the per-tick path that reads attributes
        directly off ``self.model``.
        """
        model = _make_model(_make_params())
        recorder = Recorder(model)
        assert recorder.model is model

    def test_check_returns_none_when_model_complete(self):
        """``check()`` returns None for a model with ``people`` and ``patches``.

        Given a model exposing both ``people`` and ``patches``,
        when ``check()`` is called,
        then it returns ``None`` and does not raise.

        Failure would indicate ``check`` has gained a side-effect observable
        to callers (e.g., raising), changing the harness contract.
        """
        recorder = Recorder(_make_model(_make_params()))
        assert recorder.check() is None

    def test_check_warns_when_people_missing(self, recwarn):
        """``check()`` emits a ``UserWarning`` mentioning ``people`` when absent.

        Given a model with ``patches`` but no ``people`` attribute,
        when ``check()`` is called,
        then it returns ``None`` (does not raise) and a single ``UserWarning``
        is emitted whose message contains ``'people'``.

        Failure implies the warn-on-missing contract has regressed — either
        no warning is raised (silent acceptance of a malformed model) or a
        hard error is thrown (breaking the harness's recoverable check).
        """
        params = _make_params()
        people_missing = SimpleNamespace(params=params, patches=StubFrame(np.array([1]), np.array([2])))

        assert Recorder(people_missing).check() is None

        people_warnings = [w for w in recwarn.list if "people" in str(w.message)]
        assert len(people_warnings) == 1
        assert issubclass(people_warnings[0].category, UserWarning)

    def test_check_warns_when_patches_missing(self, recwarn):
        """``check()`` emits a ``UserWarning`` mentioning ``patches`` when absent.

        Same scenario as the ``people`` case but for ``patches``. Failure
        implies an asymmetric handling of the two model frames or a
        regression in the warn-don't-raise contract.
        """
        params = _make_params()
        patches_missing = SimpleNamespace(params=params, people=StubFrame(np.array([1]), np.array([2])))

        assert Recorder(patches_missing).check() is None

        patches_warnings = [w for w in recwarn.list if "patches" in str(w.message)]
        assert len(patches_warnings) == 1
        assert issubclass(patches_warnings[0].category, UserWarning)


class TestRecorderCallGating:
    """The four-quadrant gate plus the final-tick timing in ``__call__``.

    All these tests assert *no HDF5 file is created* by checking the tmp_path
    directory after a single call. A failure of any gating test means the
    recorder is now writing under conditions where it shouldn't, which would
    produce surprise files in users' working directories.
    """

    def test_does_nothing_on_non_final_tick(self, tmp_path, monkeypatch):
        """A non-final tick never writes, even if all gate keys are set.

        Given hdf5_output=True, return=["susceptible"], and tick=0 of a
        5-tick simulation,
        when the recorder is called,
        then no ``.h5`` files are produced in the working directory.

        Failure implies the timing guard ``tick == nticks - 1`` is broken.
        """
        monkeypatch.chdir(tmp_path)
        params = _make_params(hdf5_output=True, **{"return": ["susceptible"]})
        model = _make_model(params)
        Recorder(model)(model, tick=0)
        assert list(tmp_path.glob("*.h5*")) == []

    def test_no_save_when_hdf5_output_absent(self, tmp_path, monkeypatch):
        """Missing ``hdf5_output`` in params suppresses writing.

        Given a params object without ``hdf5_output`` and a final-tick call,
        when the recorder fires,
        then no file is written.

        Failure implies the recorder is treating a missing key as truthy,
        which would write under default configs.
        """
        monkeypatch.chdir(tmp_path)
        params = _make_params(**{"return": ["susceptible"]})  # no hdf5_output key
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)
        assert list(tmp_path.glob("*.h5*")) == []

    def test_no_save_when_hdf5_output_false(self, tmp_path, monkeypatch):
        """Explicit ``hdf5_output=False`` suppresses writing.

        Given hdf5_output=False and a non-empty ``return`` list on the final
        tick, when the recorder fires, then no file is written.

        Failure implies the gate is checking only ``return`` and ignoring
        the explicit disable.
        """
        monkeypatch.chdir(tmp_path)
        params = _make_params(hdf5_output=False, **{"return": ["susceptible"]})
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)
        assert list(tmp_path.glob("*.h5*")) == []

    def test_no_save_when_return_absent(self, tmp_path, monkeypatch):
        """Missing ``return`` suppresses writing even with hdf5_output=True.

        Given hdf5_output=True but no ``return`` key on the final tick,
        when the recorder fires, then no file is written.

        Failure implies the recorder is iterating an empty property list and
        producing empty HDF5 files instead of skipping the write.
        """
        monkeypatch.chdir(tmp_path)
        params = _make_params(hdf5_output=True)  # no "return" key
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)
        assert list(tmp_path.glob("*.h5*")) == []

    def test_no_save_when_return_empty(self, tmp_path, monkeypatch):
        """An empty ``return`` list is falsy and suppresses writing.

        The gate uses ``model.params["return"]`` in a truthy context, so an
        empty list short-circuits.

        Failure implies the gate is using ``is not None`` rather than
        truthiness, which would treat ``[]`` as a valid (but empty) write.
        """
        monkeypatch.chdir(tmp_path)
        params = _make_params(hdf5_output=True, **{"return": []})
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)
        assert list(tmp_path.glob("*.h5*")) == []

    def test_save_when_hdf5_output_and_return_set(self, tmp_path, monkeypatch):
        """Both gates open on the final tick: exactly one ``.h5`` file is produced.

        Given hdf5_output=True and return=["susceptible"] on the final tick,
        when the recorder fires,
        then exactly one ``.h5`` file appears in the working directory and it
        has a timestamp-style (14-digit numeric) stem.

        Failure implies the happy-path gate is broken, the file extension is
        wrong, or the filename derivation is no longer ``<ts>.h5``.
        """
        monkeypatch.chdir(tmp_path)
        params = _make_params(hdf5_output=True, **{"return": ["susceptible"]})
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)

        files = sorted(tmp_path.glob("*.h5"))
        assert len(files) == 1
        assert files[0].stem.isdigit()
        assert len(files[0].stem) == 14


class TestRecorderCallOutputPath:
    """``outdir`` and ``compress`` shape the final filename and extension."""

    def test_outdir_param_routes_file_to_that_directory(self, tmp_path):
        """``params.outdir`` directs the output away from cwd.

        Given outdir=<tmp_path/recorder_out>, hdf5_output=True, return=["susceptible"]
        on the final tick,
        when the recorder fires,
        then the produced ``.h5`` file lives inside the outdir and the
        outdir was created on demand.

        Failure implies the recorder ignored ``outdir`` (writing to cwd) or
        crashed because the directory did not yet exist.
        """
        outdir = tmp_path / "recorder_out"
        params = _make_params(hdf5_output=True, outdir=str(outdir), **{"return": ["susceptible"]})
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)

        files = sorted(outdir.glob("*.h5"))
        assert outdir.exists()
        assert len(files) == 1

    def test_compress_param_writes_h5_gz(self, tmp_path):
        """``compress=True`` writes a gzipped HDF5 file with ``.h5.gz`` suffix.

        Given compress=True on top of the happy-path gate,
        when the recorder fires,
        then exactly one ``.h5.gz`` file appears (no plain ``.h5`` is left
        behind) and gunzipping it yields a valid HDF5 stream.

        Failure implies the compression path produced an extra side-file,
        missed the suffix change, or wrote an invalid gzip stream.
        """
        params = _make_params(hdf5_output=True, compress=True, outdir=str(tmp_path), **{"return": ["susceptible"]})
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)

        gz_files = sorted(tmp_path.glob("*.h5.gz"))
        # `glob("*.h5")` matches `.h5.gz` too on some platforms; filter to plain `.h5`.
        plain_only = [p for p in tmp_path.glob("*.h5") if p.suffix == ".h5"]
        assert len(gz_files) == 1
        assert plain_only == []

        # And the gzipped payload decompresses to a valid HDF5 file.
        with gzip.open(gz_files[0], "rb") as gz:
            with h5py.File(gz, "r") as h5file:
                assert "people" in h5file


class TestSaveHdf5:
    """Low-level ``save_hdf5`` writes correct group + dataset structure."""

    def test_creates_groups_and_datasets_for_requested_properties(self, tmp_path):
        """``people`` and ``patches`` groups exist with only the requested datasets.

        Given a model whose frames each have ``susceptible``, ``infected``,
        ``_private``, and ``helper_method``, and ``params["return"] = ["susceptible"]``,
        when ``save_hdf5_parameters`` writes the file,
        then both ``people/susceptible`` and ``patches/susceptible`` datasets
        exist and no other public properties are present in either group.

        Failure implies the return-list filter is leaking other properties
        (e.g., ``infected``) into the file or that one of the two frame groups
        is missing.
        """
        params = _make_params(**{"return": ["susceptible"]})
        model = _make_model(params)
        target = tmp_path / "out.h5"
        result = save_hdf5_parameters(model, target)
        assert result == target

        with h5py.File(target, "r") as h5file:
            assert set(h5file.keys()) == {"people", "patches"}
            assert list(h5file["people"].keys()) == ["susceptible"]
            assert list(h5file["patches"].keys()) == ["susceptible"]
            np.testing.assert_array_equal(h5file["people"]["susceptible"][:], model.people.susceptible)
            np.testing.assert_array_equal(h5file["patches"]["susceptible"][:], model.patches.susceptible)

    def test_underscore_attributes_excluded_even_if_in_return(self, tmp_path):
        """Underscore-prefixed attributes are dropped before the return filter.

        Given ``_private`` exists on both frames and ``params["return"]`` lists
        ``_private``,
        when ``save_hdf5`` runs,
        then no ``_private`` dataset is written to either group.

        Failure implies private attributes can be exposed by adding their
        names to the return list, which would bypass the protection.
        """
        params = _make_params(**{"return": ["_private", "susceptible"]})
        model = _make_model(params)
        target = tmp_path / "out.h5"
        save_hdf5_parameters(model, target)

        with h5py.File(target, "r") as h5file:
            assert "_private" not in h5file["people"]
            assert "_private" not in h5file["patches"]
            assert "susceptible" in h5file["people"]

    def test_method_attributes_excluded_even_if_in_return(self, tmp_path):
        """Bound-method attributes are dropped before the return filter.

        Given ``helper_method`` is a method on the stub frames and
        ``params["return"]`` lists ``helper_method``,
        when ``save_hdf5`` runs,
        then ``helper_method`` is not present in either group.

        Failure implies the MethodType filter is broken and methods can
        accidentally be serialized — h5py would then raise a TypeError trying
        to coerce the bound method into a dataset.
        """
        params = _make_params(**{"return": ["helper_method", "susceptible"]})
        model = _make_model(params)
        target = tmp_path / "out.h5"
        save_hdf5_parameters(model, target)

        with h5py.File(target, "r") as h5file:
            assert "helper_method" not in h5file["people"]
            assert "helper_method" not in h5file["patches"]

    def test_raises_attribute_error_when_people_missing(self, tmp_path):
        """``save_hdf5`` raises ``AttributeError`` if ``model.people`` is absent.

        Given a model without a ``people`` attribute,
        when ``save_hdf5_parameters`` is invoked,
        then ``AttributeError`` is raised mentioning ``people``.

        Failure implies the recorder silently writes a half-empty file
        (only ``patches`` group), which would corrupt downstream readers.
        """
        params = _make_params(**{"return": ["susceptible"]})
        model = SimpleNamespace(params=params, patches=StubFrame(np.array([1]), np.array([2])))
        target = tmp_path / "out.h5"
        with pytest.raises(AttributeError, match="people"):
            save_hdf5_parameters(model, target)

    def test_raises_attribute_error_when_patches_missing(self, tmp_path):
        """``save_hdf5`` raises ``AttributeError`` if ``model.patches`` is absent.

        Same as the ``people`` case, for the other frame. Failure implies an
        asymmetric handling of the two model frames.

        Note: ``people`` is iterated first, so when it succeeds and
        ``patches`` is missing, the file is partially populated before the
        raise. The partial file is acceptable behaviour for this test — we
        only assert that the exception is raised.
        """
        params = _make_params(**{"return": ["susceptible"]})
        model = SimpleNamespace(params=params, people=StubFrame(np.array([1]), np.array([2])))
        target = tmp_path / "out.h5"
        with pytest.raises(AttributeError, match="patches"):
            save_hdf5_parameters(model, target)


class TestSaveCompressedHdf5Parameters:
    """Compressed writer produces a gzipped HDF5 with the expected suffix."""

    def test_returns_path_with_gz_suffix(self, tmp_path):
        """The returned ``Path`` has ``.h5.gz`` as its concatenated suffix.

        Given a starting filename of ``out.h5``,
        when ``save_compressed_hdf5_parameters`` writes the file,
        then the returned path has name ``out.h5.gz`` and the file exists on
        disk.

        Failure implies the caller cannot trust the returned path for
        downstream logging or chaining, since the suffix add-on is silent.
        """
        params = _make_params(**{"return": ["susceptible"]})
        model = _make_model(params)
        target = tmp_path / "out.h5"
        result = save_compressed_hdf5_parameters(model, target)

        assert result.name == "out.h5.gz"
        assert result.exists()

    def test_compressed_payload_decompresses_to_valid_hdf5(self, tmp_path):
        """Gunzipping the output yields a valid HDF5 file with the expected groups.

        Given a successful compressed write,
        when the ``.h5.gz`` file is gunzipped and opened with h5py,
        then both ``people`` and ``patches`` groups are present and the
        ``susceptible`` dataset round-trips byte-for-byte.

        Failure implies the gzip step corrupted the HDF5 payload, or the
        in-memory buffer was flushed in the wrong order.
        """
        params = _make_params(**{"return": ["susceptible"]})
        model = _make_model(params)
        target = tmp_path / "snapshot.h5"
        result = save_compressed_hdf5_parameters(model, target)

        with gzip.open(result, "rb") as gz:
            with h5py.File(gz, "r") as h5file:
                assert {"people", "patches"} <= set(h5file.keys())
                np.testing.assert_array_equal(h5file["people"]["susceptible"][:], model.people.susceptible)


class TestRecorderEndToEnd:
    """``Recorder.__call__`` on the happy path produces a file we can read back."""

    def test_call_produces_readable_h5_with_expected_data(self, tmp_path):
        """End-to-end: the happy-path ``__call__`` writes data we can verify.

        Given hdf5_output=True, return=["susceptible","infected"], outdir
        pointing at tmp_path on the final tick,
        when the recorder fires,
        then the resulting ``.h5`` file contains both datasets under both
        groups with the exact stub values.

        Failure implies a regression at the integration boundary between
        ``__call__`` and ``save_hdf5_parameters`` — even if the unit tests
        of each piece pass, the wiring could be off (wrong directory,
        wrong filename pattern, wrong frame iteration).
        """
        params = _make_params(
            hdf5_output=True,
            outdir=str(tmp_path),
            **{"return": ["susceptible", "infected"]},
        )
        model = _make_model(params)
        Recorder(model)(model, tick=params.nticks - 1)

        files = sorted(Path(tmp_path).glob("*.h5"))
        assert len(files) == 1

        with h5py.File(files[0], "r") as h5file:
            for frame_name, frame in (("people", model.people), ("patches", model.patches)):
                assert {"susceptible", "infected"} == set(h5file[frame_name].keys())
                np.testing.assert_array_equal(h5file[frame_name]["susceptible"][:], frame.susceptible)
                np.testing.assert_array_equal(h5file[frame_name]["infected"][:], frame.infected)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
