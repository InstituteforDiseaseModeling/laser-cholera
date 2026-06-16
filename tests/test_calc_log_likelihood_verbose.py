"""Verbose-logging coverage tests for the calc_log_likelihood_* distributions.

Each distribution's ``verbose=True`` branch emits one or more
``logger.info(...)`` calls; these tests exercise those branches so the
verbose-only lines aren't dead in coverage. Tests assert finiteness, not
exact log content — log-text changes shouldn't cause coverage tests to fail.
"""

import contextlib
import logging

import numpy as np
import pytest

from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_beta
from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_binomial
from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_gamma
from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_normal
from laser.cholera.calc_log_likelihood_distributions import calc_log_likelihood_poisson


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


class TestDistributionVerbosePaths:
    """``verbose=True`` exercises the per-distribution INFO log branches."""

    def test_beta_verbose_logs_phi_and_total(self):
        """Beta in mean-precision mode logs ``phi`` and the total log-likelihood.

        Failure implies the verbose blocks in ``calc_log_likelihood_beta``
        are dead, leaving the user with no observability when debugging Beta
        likelihoods on bad fits.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_beta(
                np.array([0.2, 0.6, 0.4]),
                np.array([0.25, 0.55, 0.35]),
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("Beta log-likelihood" in m for m in logs)

    def test_beta_verbose_standard_shape_logs_shape_params(self):
        """Beta in standard-shape mode logs ``shape_1`` and ``shape_2``.

        Failure implies the ``mean_precision=False`` verbose branch is dead.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_beta(
                np.array([0.2, 0.6, 0.4, 0.5]),
                np.array([0.25, 0.55, 0.35, 0.45]),
                mean_precision=False,
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("Standard shape" in m or "shape_1" in m for m in logs)

    def test_binomial_verbose_logs_total(self):
        """Binomial logs the total LL when verbose.

        Failure implies the verbose branch is dead.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_binomial(
                observed=np.array([3, 4, 2]),
                estimated=np.array([0.3, 0.5, 0.25]),
                trials=np.array([10, 10, 8]),
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("Binomial" in m for m in logs)

    def test_gamma_verbose_logs_shape_and_total(self):
        """Gamma logs estimated shape and total LL.

        Failure implies the verbose branch is dead.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_gamma(
                np.array([2.5, 3.2, 1.8]),
                np.array([2.4, 3.0, 2.0]),
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("Gamma" in m for m in logs)

    def test_normal_verbose_logs_sigma_and_total(self):
        """Normal logs estimated sigma and total LL.

        Failure implies the verbose branch is dead. The Shapiro-Wilk
        message also fires here; we don't assert on it because the p-value
        depends on RNG.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_normal(
                np.array([1.2, 2.8, 3.1, 4.0]),
                np.array([1.0, 3.0, 3.2, 4.1]),
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("Normal log-likelihood" in m or "σ" in m for m in logs)

    def test_poisson_verbose_logs_total(self):
        """Poisson logs the total LL when verbose.

        Failure implies the verbose branch is dead.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_poisson(
                np.array([2, 3, 4], dtype=float),
                np.array([2.2, 2.9, 4.1]),
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("Poisson" in m for m in logs)

    def test_poisson_all_zero_observed_logs_info(self):
        """Poisson logs the 'All observations are zero' branch.

        Failure implies the diagnostic for fully-zero observed data has
        regressed — users would silently get a 0 log-likelihood with no
        signal that the data is degenerate.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_poisson(
                np.zeros(4),
                np.array([1.0, 1.0, 1.0, 1.0]),
                verbose=True,
            )
        assert np.isfinite(ll)
        # The function logs "All observations are zero (or NA)."
        assert any("zero" in m.lower() for m in logs)

    def test_poisson_verbose_logs_proportional_penalty(self):
        """Poisson with zero estimate + nonzero observed logs the penalty branch.

        Failure implies the verbose log inside the penalty branch
        (calc_log_likelihood_distributions.py lines ~651-656) is dead.
        """
        with _capture_logs() as logs:
            ll = calc_log_likelihood_poisson(
                np.array([5, 3], dtype=float),
                np.array([0.0, 3.0]),
                zero_buffer=False,
                verbose=True,
            )
        assert np.isfinite(ll)
        assert any("proportional penalty" in m.lower() for m in logs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
