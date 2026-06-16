"""Tests for laser.cholera.metapop.analyzer.

The Analyzer is a per-tick hook whose only behavior of interest is the final-tick
log-likelihood computation, gated on ``params.calc_likelihood``. These tests
exercise the gate, the optional-kwargs passthrough into
``calc_model_likelihood``, and the ``ValueError`` recovery path that yields
``-np.inf`` rather than propagating to the caller.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from laser.cholera.metapop import analyzer as analyzer_module
from laser.cholera.metapop.analyzer import Analyzer
from laser.cholera.metapop.params import PropertySetEx


def _make_params(**kwargs) -> PropertySetEx:
    """Build a minimal PropertySetEx with the keys ``Analyzer.__call__`` reads.

    ``nticks=4`` makes the final tick index 3. ``reported_cases`` /
    ``reported_deaths`` are shape (n_locations=1, n_time_steps=4). Other keys
    are opt-in so each test can target a specific code path.
    """
    payload = {
        "nticks": 4,
        "reported_cases": np.array([[5.0, 6.0, 7.0, 8.0]], dtype=float),
        "reported_deaths": np.array([[0.5, 0.6, 0.7, 0.8]], dtype=float),
    }
    payload.update(kwargs)
    return PropertySetEx(payload)


def _make_model(params: PropertySetEx, n_locations: int = 1) -> SimpleNamespace:
    """Compose a stub model with the three attribute trees Analyzer reads.

    The shapes line up so the ``nreports`` calculation in ``__call__`` returns
    something sensible: ``incidence.shape[0] - 1`` and
    ``reported_cases.shape[1]`` both equal the number of time steps.
    """
    n_time = params.reported_cases.shape[1]
    results = SimpleNamespace(
        reported_cases=np.array([[5.0, 6.0, 7.0, 8.0]], dtype=float),
        reported_deaths=np.array([[0.5, 0.6, 0.7, 0.8]], dtype=float),
    )
    patches = SimpleNamespace(
        # +1 row so the analyzer's `incidence.shape[0] - 1` matches reported_cases width
        incidence=np.zeros((n_time + 1, n_locations), dtype=float),
    )
    return SimpleNamespace(params=params, results=results, patches=patches)


class TestAnalyzerBasics:
    """Construction and ``check`` behavior.

    Failure of these tests implies either ``__init__`` is not storing the model
    reference or ``check`` has acquired side-effects it should not have.
    """

    def test_init_stores_model_reference(self):
        """``Analyzer(model).model`` is identity-equal to the model passed in.

        Given any stub model, when an Analyzer is constructed, then its
        ``.model`` attribute is the exact same object.

        Failure implies the constructor has begun copying the model.
        """
        model = _make_model(_make_params())
        analyzer = Analyzer(model)
        assert analyzer.model is model

    def test_check_returns_none(self):
        """``check()`` returns ``None`` and does not raise.

        Given an Analyzer wrapping any stub model, when ``check()`` is called,
        then it returns ``None`` without side-effects.

        Failure implies ``check`` has acquired observable behavior that
        callers don't currently expect.
        """
        analyzer = Analyzer(_make_model(_make_params()))
        assert analyzer.check() is None


class TestAnalyzerCallGating:
    """The final-tick gate plus the ``calc_likelihood`` guard."""

    def test_non_final_tick_does_nothing(self):
        """An early tick neither sets ``log_likelihood`` nor invokes the LL call.

        Given a 4-tick simulation with ``calc_likelihood=True`` and tick=0,
        when the analyzer fires, then the model has no ``log_likelihood``
        attribute afterwards.

        Failure implies the timing guard ``tick == nticks - 1`` is broken
        and the LL would be computed mid-simulation, wasting work.
        """
        params = _make_params(calc_likelihood=True)
        model = _make_model(params)
        Analyzer(model)(model, tick=0)
        assert not hasattr(model, "log_likelihood")

    def test_final_tick_with_calc_likelihood_absent_sets_nan(self):
        """Missing ``calc_likelihood`` on the final tick sets ``log_likelihood`` to NaN.

        Given a final tick and no ``calc_likelihood`` key in params, when the
        analyzer fires, then ``model.log_likelihood`` is NaN (the explicit
        "not computed" sentinel).

        Failure implies the analyzer either crashes on a missing key or
        defaults to a non-NaN value, which would confuse downstream tooling.
        """
        params = _make_params()  # no calc_likelihood
        model = _make_model(params)
        Analyzer(model)(model, tick=params.nticks - 1)
        assert hasattr(model, "log_likelihood")
        assert np.isnan(model.log_likelihood)

    def test_final_tick_with_calc_likelihood_false_sets_nan(self):
        """``calc_likelihood=False`` on the final tick also sets ``log_likelihood`` to NaN.

        Given ``calc_likelihood=False`` on the final tick, the gate must not
        fire; the explicit-not-computed NaN sentinel is set instead.

        Failure implies the gate is checking presence-only and ignoring the
        explicit disable.
        """
        params = _make_params(calc_likelihood=False)
        model = _make_model(params)
        Analyzer(model)(model, tick=params.nticks - 1)
        assert np.isnan(model.log_likelihood)

    def test_final_tick_with_calc_likelihood_true_sets_finite(self):
        """``calc_likelihood=True`` on the final tick sets a finite log-likelihood.

        Given matched obs/est arrays and ``calc_likelihood=True`` on the final
        tick, when the analyzer fires, then ``model.log_likelihood`` is a
        finite scalar.

        Failure implies the happy-path wiring between Analyzer and
        ``calc_model_likelihood`` is broken or producing NaN/Inf.
        """
        params = _make_params(calc_likelihood=True)
        model = _make_model(params)
        Analyzer(model)(model, tick=params.nticks - 1)
        assert np.isfinite(model.log_likelihood)


class TestAnalyzerCallPassthrough:
    """Optional kwargs in params flow through to ``calc_model_likelihood``."""

    def test_optional_kwargs_pass_through_to_calc_model_likelihood(self, monkeypatch):
        """Every recognized params key reaches ``calc_model_likelihood`` as a kwarg.

        Given params containing several recognized optional keys
        (``weight_peak_timing``, ``epidemic_peaks``, ``date_start``), when the
        analyzer fires on the final tick, then the underlying
        ``calc_model_likelihood`` is invoked with each of those keys present
        in its kwargs (and absent params keys are not introduced).

        Inconsistency note: the analyzer iterates a hard-coded allowlist of
        recognized keys (see analyzer.py:24-43). New optional kwargs added to
        ``calc_model_likelihood`` must also be added to that list. This test
        pins the passthrough mechanism, not the full allowlist.

        Failure implies the kwargs filter is dropping recognized keys, which
        would silently disable shape-term weights set in the params file.
        """
        captured = {}

        def fake_calc(**kwargs):
            captured.update(kwargs)
            return -42.0

        monkeypatch.setattr(analyzer_module, "calc_model_likelihood", fake_calc)

        params = _make_params(
            calc_likelihood=True,
            weight_peak_timing=0.25,
            epidemic_peaks="sentinel_value",
            date_start="2024-01-01",
        )
        model = _make_model(params)
        Analyzer(model)(model, tick=params.nticks - 1)

        assert model.log_likelihood == -42.0
        for key in ("weight_peak_timing", "epidemic_peaks", "date_start"):
            assert key in captured, f"{key} not forwarded to calc_model_likelihood"
        assert captured["weight_peak_timing"] == 0.25
        assert captured["epidemic_peaks"] == "sentinel_value"

    def test_value_error_in_underlying_call_sets_neg_inf(self, monkeypatch):
        """A ``ValueError`` raised inside ``calc_model_likelihood`` becomes ``-np.inf``.

        Given a final-tick call where the underlying LL implementation raises
        ``ValueError`` (e.g., shape mismatch on real-world inputs), when the
        analyzer fires, then ``model.log_likelihood`` is set to ``-np.inf``
        instead of propagating the exception.

        Failure implies the recovery path has regressed and a configuration
        error in any params key would now crash the entire simulation
        post-loop, rather than producing a worst-case score that downstream
        calibration tooling can compare.
        """

        def boom(**_kwargs):
            raise ValueError("synthetic shape mismatch")

        monkeypatch.setattr(analyzer_module, "calc_model_likelihood", boom)

        params = _make_params(calc_likelihood=True)
        model = _make_model(params)
        Analyzer(model)(model, tick=params.nticks - 1)
        assert model.log_likelihood == -np.inf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
