"""Tests for calc_log_likelihood_beta() — Beta-distribution log-likelihood helper.

Pinned with hand-computed expected values where possible. Beta has two modes
(mean-precision and standard shape); both branches are exercised.
"""

import contextlib
import logging

import numpy as np
import pytest

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_beta


@contextlib.contextmanager
def _capture_logs(level=logging.INFO):
    logger = logging.getLogger("laser.cholera")
    prev_level = logger.level
    logger.setLevel(level)
    records = []

    class _Handler(logging.Handler):
        def emit(self, record):
            records.append(f"{record.levelname}:{record.name}:{record.getMessage()}")

    handler = _Handler()
    logger.addHandler(handler)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)


class TestCalcLogLikelihoodBeta:
    """Tests for ``calc_log_likelihood_beta``."""

    def test_errors_when_observed_and_estimated_lengths_differ(self):
        """Length mismatch raises with the expected message.

        Failure implies the function silently proceeds on mismatched arrays.
        """
        with pytest.raises(ValueError, match="Lengths of observed and estimated must match"):
            calc_log_likelihood_beta(np.array([0.2, 0.6]), np.array([0.3]), verbose=False)

    def test_returns_nan_for_all_na_input(self):
        """All-NaN input returns NaN and logs 'No usable data' under verbose.

        Failure implies the no-data path returns a numeric value or stays silent.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_beta(
                np.array([np.nan, np.nan]),
                np.array([np.nan, np.nan]),
                verbose=True,
            )
        assert np.isnan(ll)
        assert any("No usable data" in m for m in logs)

    def test_errors_on_negative_weights(self):
        """Negative weights raise.

        Failure implies the weighted-sum semantics are broken.
        """
        with pytest.raises(ValueError, match="All weights must be >= 0"):
            calc_log_likelihood_beta(
                np.array([0.2, 0.5]),
                np.array([0.25, 0.55]),
                weights=np.array([1, -1], dtype=float),
                verbose=False,
            )

    def test_errors_on_zero_sum_weights(self):
        """All-zero weights raise.

        Failure implies a divide-by-zero hides a configuration error.
        """
        with pytest.raises(ValueError, match="All weights are zero"):
            calc_log_likelihood_beta(
                np.array([0.2, 0.5]),
                np.array([0.25, 0.55]),
                weights=np.array([0, 0], dtype=float),
                verbose=False,
            )

    def test_errors_on_observed_outside_unit_interval(self):
        """Observed values <= 0 or >= 1 raise — Beta is undefined there.

        Failure implies the function would emit -inf or NaN silently for
        out-of-domain inputs.
        """
        with pytest.raises(ValueError, match="observed must be strictly between 0 and 1"):
            calc_log_likelihood_beta(np.array([0.5, 1.5]), np.array([0.4, 0.6]), verbose=False)

    def test_errors_on_estimated_outside_unit_interval(self):
        """Estimated values <= 0 or >= 1 raise.

        Failure implies bad model output silently produces NaN.
        """
        with pytest.raises(ValueError, match="estimated must be strictly between 0 and 1"):
            calc_log_likelihood_beta(np.array([0.4, 0.6]), np.array([0.5, 1.5]), verbose=False)

    def test_returns_finite_for_typical_input(self):
        """Typical proportion data returns a finite scalar.

        Failure implies the happy-path computation is broken.
        """
        ll = calc_log_likelihood_beta(
            np.array([0.2, 0.6, 0.4]),
            np.array([0.25, 0.55, 0.35]),
            verbose=False,
        )
        assert np.isfinite(ll)

    def test_mean_precision_mode_matches_docstring_value(self):
        """Default mean-precision call matches the value pinned in the docstring example.

        Failure implies the mean-precision parameterization has shifted.
        """
        ll = calc_log_likelihood_beta(
            np.array([0.2, 0.6, 0.4]),
            np.array([0.25, 0.55, 0.35]),
            verbose=False,
        )
        assert ll == pytest.approx(4.770704709814893, rel=1e-9)

    def test_standard_shape_mode_returns_finite(self):
        """``mean_precision=False`` exercises the shape-estimation branch and returns finite.

        Failure implies the standard-shape branch is dead or broken.
        """
        ll = calc_log_likelihood_beta(
            np.array([0.2, 0.6, 0.4, 0.5]),
            np.array([0.25, 0.55, 0.35, 0.45]),
            mean_precision=False,
            verbose=False,
        )
        assert np.isfinite(ll)

    def test_errors_on_flat_observed_in_standard_shape_mode(self):
        """Flat observed (zero variance) raises in standard-shape mode — shape params undefined.

        Failure implies the flat-input guard has regressed.
        """
        with pytest.raises(ValueError, match="Observed variance"):
            calc_log_likelihood_beta(
                np.array([0.3, 0.3, 0.3]),
                np.array([0.3, 0.3, 0.3]),
                mean_precision=False,
                verbose=False,
            )

    def test_weights_scale_the_log_likelihood(self):
        """Doubling the weights doubles the resulting log-likelihood (linear in weights).

        Failure implies weights are not applied to the per-element log-pdf.
        """
        obs = np.array([0.2, 0.6, 0.4])
        est = np.array([0.25, 0.55, 0.35])
        ll1 = calc_log_likelihood_beta(obs, est, verbose=False)
        ll2 = calc_log_likelihood_beta(obs, est, weights=np.array([2.0, 2.0, 2.0]), verbose=False)
        assert ll2 == pytest.approx(2 * ll1, rel=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
