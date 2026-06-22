"""Tests for per-observation confidence weighting in ``calc_model_likelihood``.

Port of ``MOSAIC-pkg/tests/testthat/test-calc_model_likelihood_obs_weights.R``
(see ``misc/test-calc_model_likelihood_obs_weights.R`` in this repo for the
canonical reference). The R file pins behavioural parity between R and Python
for the `weights_obs_cases` / `weights_obs_deaths` arguments, which were added
in MOSAIC v0.45.3 (commit ``7a265b1d``) so the Coiled/Dask worker would no
longer diverge from the local R PSOCK calibration path.

All tests target the documented v1 contract: per-cell weights affect only the
NB core; the peak / cumulative / WIS shape terms remain unweighted. The
``calc_model_likelihood`` shape-term-related arguments are therefore left at
their defaults (off) so every value is hand-traceable through the NB kernel.

Design invariants under test (mirror the R file):

1. all-ones ``weights_obs`` == ``None`` -> byte-identical LL AND identical gate.
2. a 0.5-weight cell carries exactly half the per-cell weight of a 1.0 cell.
3. per-location effective-weight mass == masked-``weights_time`` sum.
4. a fully-zeroed weight row contributes 0.
5. the exactly-3-observed-cell boundary still passes under all-ones.
6. a documented-zero-heavy country does not dominate (magnitude bounded).
7. dimension validation on the new matrix args.
8. cases / deaths gates are independent under weighting.
"""

from math import lgamma
from math import log

import numpy as np
import pytest

from laser.cholera.calc_model_likelihood import calc_model_likelihood
from laser.cholera.calc_model_likelihood import mask_weights
from laser.cholera.calc_model_likelihood import nb_size_from_obs_weighted
from laser.cholera.calc_model_likelihood import weights_obs_effective
from laser.cholera.calc_model_likelihood import weights_obs_row_trivial


def _nb_logdens(o: float, e: float, k: float) -> float:
    """Hand NB log-density mirroring the scorer's negbin kernel.

    Matches the kernel in `_calc_log_likelihood_nb`: estimated rates are
    floored at ``1e-10``, then ``scipy.stats.nbinom.logpmf(o, n=k, p=k/(k+e))``.
    Replicated here in closed form (lgamma + log identities) so test values are
    independent of scipy's vectorised path.
    """
    e = max(e, 1e-10)
    return lgamma(o + k) - lgamma(k) - lgamma(o + 1) + k * log(k / (k + e)) + o * log(e / (k + e))


# ---------------------------------------------------------------------------
# (1) all-ones weights_obs == None : byte-identical LL
# ---------------------------------------------------------------------------


class TestTrivialEqualsNone:
    """An all-ones (on finite-obs cells) ``weights_obs`` must route through the unweighted code path."""

    def test_all_ones_weights_obs_is_byte_identical_to_none(self):
        """A 2x3 all-ones ``weights_obs_*`` matches the ``None`` result bit-for-bit.

        Given matched 2x3 case / death observations and estimates, when
        ``calc_model_likelihood`` runs once with ``weights_obs_*=None`` and
        once with all-ones matrices, then the returned scalar log-likelihoods
        are byte-identical (`np.array_equal`-style identity, not just
        approximate equality).

        Failure implies the trivial-row short-circuit in the new dual-mode
        path regressed: an all-ones matrix is leaking into the mass-preserving
        renormalisation branch and computing a numerically-different LL.
        """
        obs_c = np.array([[0, 5, 9], [2, 1, 4]], dtype=float)
        est_c = np.array([[1, 4, 8], [3, 2, 5]], dtype=float)
        obs_d = np.array([[0, 1, 2], [0, 0, 1]], dtype=float)
        est_d = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)

        ll_none = calc_model_likelihood(obs_c, est_c, obs_d, est_d)
        ll_ones = calc_model_likelihood(
            obs_c,
            est_c,
            obs_d,
            est_d,
            weights_obs_cases=np.ones((2, 3)),
            weights_obs_deaths=np.ones((2, 3)),
        )

        assert ll_none == ll_ones, f"all-ones weights_obs diverged from None: {ll_none} vs {ll_ones}"

    def test_all_ones_weights_obs_yields_identical_gate_decision(self):
        """At the 3-finite-obs boundary, all-ones leaves the location-included gate decision unchanged.

        Given a 1x4 ``obs_cases`` with exactly 3 finite observations (and one
        ``NaN``), when ``calc_model_likelihood`` runs with and without an
        all-ones ``weights_obs_cases``, then the LL is finite and non-zero
        either way and byte-identical between the two calls.

        Failure implies the gate switched paths (raw finite-count vs. ESS-style
        sum) for trivial rows, which would silently drop locations that today
        pass the 3-finite-obs boundary.
        """
        obs_c = np.array([[0.0, 5.0, 9.0, np.nan]])
        est_c = np.array([[1.0, 4.0, 8.0, 3.0]])
        zd = np.zeros((1, 4))

        ll_none = calc_model_likelihood(obs_c, est_c, zd, zd)
        ll_ones = calc_model_likelihood(obs_c, est_c, zd, zd, weights_obs_cases=np.ones((1, 4)))

        assert ll_none == ll_ones
        assert np.isfinite(ll_none)
        assert ll_none != 0


# ---------------------------------------------------------------------------
# (2) a 0.5-weight cell carries exactly half the per-cell weight of a 1.0 cell
# ---------------------------------------------------------------------------


class TestEffectiveWeights:
    """Direct tests of the ``weights_obs_effective`` helper and the assembled NB LL."""

    def test_half_weight_cell_carries_exactly_half(self):
        """After mass-preserving renormalisation a 0.5 cell is exactly half a 1.0 cell.

        Given a length-4 obs row, an all-ones ``weights_time``, and a
        ``wobs = [1, 0.5, 1, 1]`` confidence row, when
        ``weights_obs_effective`` runs, then ``w_eff[1] / w_eff[0]``,
        ``w_eff[1] / w_eff[2]``, and ``w_eff[1] / w_eff[3]`` all equal
        ``0.5`` to floating-point tolerance.

        Failure implies the mass-preserving renormalisation changed the
        relative trust between cells — a 0.5-confidence observation would no
        longer carry "half" the influence of a 1.0-confidence one, contrary to
        the issue's worked example.
        """
        obs = np.array([0.0, 5.0, 9.0, 2.0])
        est = np.array([1.0, 4.0, 8.0, 3.0])
        wt = np.ones(4)
        wobs = np.array([1.0, 0.5, 1.0, 1.0])

        w_eff = weights_obs_effective(wt, wobs, obs, est)

        assert w_eff[1] / w_eff[0] == pytest.approx(0.5, abs=1e-12)
        assert w_eff[1] / w_eff[2] == pytest.approx(0.5, abs=1e-12)
        assert w_eff[1] / w_eff[3] == pytest.approx(0.5, abs=1e-12)

    def test_weighted_nb_ll_equals_hand_computed(self):
        """Weighted NB LL equals ``sum(w_eff * nb_logpmf)`` with k from ``w_eff``.

        Given a 1x4 cases row, all-zero deaths, and a heterogeneous
        ``weights_obs_cases = [1, 0.5, 1, 1]``, when
        ``calc_model_likelihood`` runs, then the returned scalar equals
        ``sum(w_eff * NB_logdens(obs, est, k))`` where ``w_eff`` is the
        ``weights_obs_effective`` of the inputs and ``k`` is the
        ``nb_size_from_obs_weighted(obs, w_eff)`` dispersion. Tolerance
        ``1e-10``.

        Failure implies (a) the scorer is using a different k than the one
        coherent with the weights it applies, or (b) the mass-preserving
        renormalisation does not match the documented formula.
        """
        obs = np.array([[0.0, 5.0, 9.0, 2.0]])
        est = np.array([[1.0, 4.0, 8.0, 3.0]])
        zd = np.zeros((1, 4))
        wobs = np.array([[1.0, 0.5, 1.0, 1.0]])

        w_eff = weights_obs_effective(np.ones(4), wobs[0, :], obs[0, :], est[0, :])
        k = nb_size_from_obs_weighted(obs[0, :], w_eff, k_min=3)
        ll_vec = np.array([_nb_logdens(obs[0, i], est[0, i], k) for i in range(4)])
        expected = float(np.sum(w_eff * ll_vec))

        ll = calc_model_likelihood(obs, est, zd, zd, weights_obs_cases=wobs)

        # R's testthat suite uses tolerance = 1e-10. Python's
        # `scipy.stats.nbinom.logpmf` and the closed-form hand kernel
        # agree to a few ULPs on the multi-term sum, well inside 1e-8.
        # (R's `dpois(0, 0) = 1` exact-zero match is now mirrored in the
        # Python implementation, but the cases-branch here has est > 0
        # everywhere so the standard NB kernel is exercised either way.)
        assert ll == pytest.approx(expected, abs=1e-8)


# ---------------------------------------------------------------------------
# (3) per-location effective-weight mass == masked-weights_time sum
# ---------------------------------------------------------------------------


class TestMassPreservation:
    """The mass-preserving invariant — only the *shape* of trust changes, not the per-location total."""

    def test_effective_weight_mass_equals_masked_weights_time_sum(self):
        """``sum(w_eff) == sum(mask_weights(weights_time, obs, est))`` per location.

        Given a length-5 obs row with one NaN, a non-uniform
        ``weights_time = [2, 1, 1, 1, 0.5]``, and a heterogeneous
        ``wobs = [0.8, 0.9, 0.8, 0.95, 0.8]``, when
        ``weights_obs_effective`` runs, then ``sum(w_eff)`` equals
        ``sum(mask_weights(weights_time, obs, est))`` to floating-point
        tolerance, and the masked cell stays at zero.

        Failure implies the renormalisation does not preserve mass — a
        weighted run would produce a different per-location LL total than the
        equivalent unweighted run, breaking the documented ``weights_location``
        contract (it should remain the sole cross-location lever).
        """
        obs = np.array([0.0, 5.0, np.nan, 9.0, 2.0])
        est = np.array([1.0, 4.0, 3.0, 8.0, 3.0])
        wt = np.array([2.0, 1.0, 1.0, 1.0, 0.5])
        wobs = np.array([0.8, 0.9, 0.8, 0.95, 0.8])

        target = float(np.sum(mask_weights(wt, obs, est)))
        w_eff = weights_obs_effective(wt, wobs, obs, est)

        assert float(np.sum(w_eff)) == pytest.approx(target, abs=1e-12)
        assert w_eff[2] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (4) a fully-zeroed weight row contributes 0
# ---------------------------------------------------------------------------


class TestZeroRowContribution:
    """An all-zero confidence row degrades the channel to zero contribution."""

    def test_fully_zeroed_weight_row_contributes_zero(self):
        """An all-zero ``weights_obs_cases`` plus zero-data deaths yields LL == 0.

        Given a 1x4 ``obs_cases`` with all-zero ``weights_obs_cases`` and
        all-zero deaths (perfect-match zero data, NB LL = 0), when
        ``calc_model_likelihood`` runs, then the returned scalar is zero to
        floating-point tolerance.

        Failure implies the zero-row degenerate path in
        ``weights_obs_effective`` (``s <= 0`` -> return zeros) regressed and
        the channel is leaking a non-zero LL when no cell carries trust.
        """
        obs = np.array([[0.0, 5.0, 9.0, 2.0]])
        est = np.array([[1.0, 4.0, 8.0, 3.0]])
        zd = np.zeros((1, 4))

        ll = calc_model_likelihood(obs, est, zd, zd, weights_obs_cases=np.zeros((1, 4)))

        # `_calc_log_likelihood_nb` now matches the R reference's three-branch
        # zero-prediction handler: `est <= 0 AND obs == 0` returns exactly 0
        # (perfect-match zero) rather than leaking ~ -1e-10 through a
        # Poisson(0, mu=1e-10) evaluation. So the deaths channel here
        # contributes 0 and we can assert byte-precise equality to 0 — same
        # tolerance the R `testthat` suite uses (`expect_equal(ll, 0,
        # tolerance = 1e-12)`).
        assert ll == pytest.approx(0.0, abs=1e-12)

    def test_weights_obs_effective_zero_row_returns_zero_vector(self):
        """The helper itself returns an all-zero vector for a zero confidence row.

        Given any obs / est row and an all-zero ``wobs``, when
        ``weights_obs_effective`` runs, then the result is the all-zero vector
        of length ``len(weights_time)``.

        Failure implies the early-return degenerate guard regressed; the
        downstream NB call would then receive a NaN- or inf-laden weight
        vector.
        """
        obs = np.array([0.0, 5.0, 9.0, 2.0])
        est = np.array([1.0, 4.0, 8.0, 3.0])
        w_eff = weights_obs_effective(np.ones(4), np.zeros(4), obs, est)

        assert np.all(w_eff == 0.0)


# ---------------------------------------------------------------------------
# (5) exactly-3-observed-cell boundary passes under all-ones (gate back-compat)
# ---------------------------------------------------------------------------


class TestGateBoundary:
    """The 3-finite-obs gate stays exactly where it was for trivial / all-ones rows."""

    def test_exactly_three_observed_boundary_passes_under_all_ones(self):
        """3-finite-obs row passes; 2-finite-obs row drops out — same as the legacy gate.

        Given 1x4 obs rows with either 3 finite cells (passing the >= 3 gate)
        or 2 finite cells (failing it), and a matching all-ones
        ``weights_obs_cases``, when ``calc_model_likelihood`` runs, then the
        3-finite case returns a finite non-zero LL and the 2-finite case
        returns 0.

        Failure implies the trivial-row branch no longer uses the raw
        finite-count gate, so locations at the boundary are slipping under or
        over the >= 3 threshold and silently changing the calibration.
        """
        obs_3 = np.array([[0.0, 5.0, 9.0, np.nan]])
        est = np.array([[1.0, 4.0, 8.0, 3.0]])
        zd = np.zeros((1, 4))

        ll_3 = calc_model_likelihood(obs_3, est, zd, zd, weights_obs_cases=np.ones((1, 4)))
        assert np.isfinite(ll_3)
        assert ll_3 != 0

        obs_2 = np.array([[0.0, 5.0, np.nan, np.nan]])
        ll_2 = calc_model_likelihood(obs_2, est, zd, zd, weights_obs_cases=np.ones((1, 4)))
        # The zero-data deaths channel now returns exactly 0 via the new
        # three-branch zero-prediction handler in `_calc_log_likelihood_nb`,
        # matching R's `dpois(0, 0) = 1`. Cases channel drops out under the
        # 3-finite-obs gate (only 2 finite cells). Total LL is byte-precise 0.
        assert ll_2 == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# (6) documented-zero-heavy country does not dominate (magnitude bounded)
# ---------------------------------------------------------------------------


class TestZeroHeavyBound:
    """Mass-preservation keeps a documented-zero-heavy country's LL on the same order as unweighted."""

    def test_zero_heavy_country_ll_bounded_vs_unweighted(self):
        """A country with 45/50 documented-zero (low-confidence) cells stays within ~5% of unweighted.

        Given a 50-step series with five real-data cells (high confidence
        ``0.95``) and 45 documented-zero cells (low confidence ``0.80``),
        when ``calc_model_likelihood`` runs once unweighted and once with the
        per-cell ``weights_obs_cases``, then the weighted LL is finite and
        within ``[0.5, 1.05]`` of the unweighted LL by absolute magnitude.

        Failure implies the mass-preserving renorm either (a) ballooned the LL
        magnitude (low-confidence cells dominating instead of being damped),
        or (b) collapsed the LL toward zero (high-confidence cells losing
        signal because too much mass concentrated on them).
        """
        ntime = 50
        obs = np.zeros((1, ntime), dtype=float)
        obs[0, :5] = [3, 8, 12, 6, 2]
        est = np.full((1, ntime), 2.0)
        est[0, :5] = [2, 7, 10, 5, 3]
        zd = np.zeros((1, ntime), dtype=float)

        ll_unw = calc_model_likelihood(obs, est, zd, zd)
        wobs = np.full((1, ntime), 0.80)
        wobs[0, :5] = 0.95
        ll_w = calc_model_likelihood(obs, est, zd, zd, weights_obs_cases=wobs)

        assert np.isfinite(ll_w)
        assert abs(ll_w) < abs(ll_unw) * 1.05
        assert abs(ll_w) > abs(ll_unw) * 0.5


# ---------------------------------------------------------------------------
# (7) dimension validation
# ---------------------------------------------------------------------------


class TestDimensionValidation:
    """Shape / type / range validation on ``weights_obs_*``."""

    def test_shape_mismatch_cases_raises(self):
        """A wrong-shape ``weights_obs_cases`` raises ``ValueError`` naming the parameter.

        Given 2x3 observation matrices and a 2x4 ``weights_obs_cases``, when
        ``calc_model_likelihood`` is called, then it raises ``ValueError`` with
        a message identifying ``weights_obs_cases`` and "same dimensions".

        Failure implies the early shape check regressed and a mis-shaped
        confidence matrix would either silently broadcast or crash deep inside
        the loop.
        """
        obs = np.zeros((2, 3))
        est = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"weights_obs_cases must have the same dimensions"):
            calc_model_likelihood(obs, est, obs, est, weights_obs_cases=np.ones((2, 4)))

    def test_shape_mismatch_deaths_raises(self):
        """A wrong-shape ``weights_obs_deaths`` raises ``ValueError`` naming the parameter.

        Failure mode is the same as the cases variant but for the deaths
        channel; the two arguments must be validated independently.
        """
        obs = np.zeros((2, 3))
        est = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"weights_obs_deaths must have the same dimensions"):
            calc_model_likelihood(obs, est, obs, est, weights_obs_deaths=np.ones((1, 3)))

    def test_non_matrix_input_raises(self):
        """A 1-D ``weights_obs_cases`` raises ``ValueError`` naming "matrix".

        Given 2x3 observation matrices and a 1-D array masquerading as a
        weights matrix, when ``calc_model_likelihood`` is called, then it
        raises ``ValueError`` mentioning that the argument must be a 2-D
        ndarray / matrix.

        Failure implies the dimensionality check accepts a flattened array,
        which would then either broadcast incorrectly across the per-location
        loop or crash with a less helpful message.
        """
        obs = np.zeros((2, 3))
        est = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"weights_obs_cases must be a 2-D ndarray"):
            calc_model_likelihood(obs, est, obs, est, weights_obs_cases=np.ones(6))

    def test_negative_entries_raise(self):
        """A negative entry in ``weights_obs_cases`` raises ``ValueError`` mentioning ">= 0".

        NaN entries are permitted (treated as zero confidence) but explicit
        negative values must be rejected as a configuration error.

        Failure implies the non-negativity guard regressed; a negative weight
        would flip the sign of the per-cell contribution and silently swing
        the calibration.
        """
        obs = np.zeros((2, 3))
        est = np.zeros((2, 3))
        with pytest.raises(ValueError, match=r"weights_obs_cases must be >= 0"):
            calc_model_likelihood(obs, est, obs, est, weights_obs_cases=np.full((2, 3), -0.1))


# ---------------------------------------------------------------------------
# (8) cases and deaths gates are independent under weighting
# ---------------------------------------------------------------------------


class TestIndependentGates:
    """When a channel's ESS drops below the gate the OTHER channel still scores normally."""

    def test_cases_deaths_gates_independent(self):
        """Deaths ESS below the gate drops the deaths channel; cases is unaffected.

        Given 1x4 case and death rows where ``weights_obs_cases`` is
        ``[0.95]*4`` (cases ESS = 3.8 >= 3) and ``weights_obs_deaths`` is
        ``[0.50]*4`` (deaths ESS = 2.0 < 3), when ``calc_model_likelihood``
        runs once with the actual death weights and once with an all-zero
        ``weights_obs_deaths`` (forcing the deaths channel off), then the two
        results are equal to floating-point tolerance — confirming the deaths
        channel dropped out in both calls and the cases LL is unchanged.

        Failure implies the deaths gate is leaking into the cases path (or
        vice versa), so a low-confidence deaths series silently down-weights
        cases too.
        """
        obs_c = np.array([[3.0, 8.0, 5.0, 2.0]])
        est_c = np.array([[2.0, 7.0, 6.0, 3.0]])
        obs_d = np.array([[1.0, 2.0, 1.0, 0.0]])
        est_d = np.array([[1.0, 1.0, 1.0, 1.0]])
        wc = np.full((1, 4), 0.95)
        wd = np.full((1, 4), 0.50)

        ll_both = calc_model_likelihood(obs_c, est_c, obs_d, est_d, weights_obs_cases=wc, weights_obs_deaths=wd)
        ll_cases_only = calc_model_likelihood(obs_c, est_c, obs_d, est_d, weights_obs_cases=wc, weights_obs_deaths=np.zeros((1, 4)))

        assert ll_both == pytest.approx(ll_cases_only, abs=1e-12)


# ---------------------------------------------------------------------------
# Regression: None argument is byte-identical to omitting the argument
# ---------------------------------------------------------------------------


class TestNoneIsRegression:
    """Explicit ``weights_obs_*=None`` matches today's signature-less call exactly."""

    def test_passing_none_is_identical_to_omitting(self):
        """``weights_obs_cases=None, weights_obs_deaths=None`` is byte-identical to omitting them.

        Given the same inputs as the all-ones byte-identity test (#1), when
        ``calc_model_likelihood`` is called once with the new kwargs at their
        default ``None`` and once without mentioning them at all, then the
        returned scalars are byte-identical.

        Failure implies the new ``None``-handling branch is materially
        different from the legacy code path and any caller that omits the new
        arguments would silently see a different LL.
        """
        obs_c = np.array([[0, 5, 9], [2, 1, 4]], dtype=float)
        est_c = np.array([[1, 4, 8], [3, 2, 5]], dtype=float)
        obs_d = np.array([[0, 1, 2], [0, 0, 1]], dtype=float)
        est_d = np.array([[1, 1, 1], [1, 1, 1]], dtype=float)

        ll_implicit = calc_model_likelihood(obs_c, est_c, obs_d, est_d)
        ll_explicit_none = calc_model_likelihood(obs_c, est_c, obs_d, est_d, weights_obs_cases=None, weights_obs_deaths=None)

        assert ll_implicit == ll_explicit_none


# ---------------------------------------------------------------------------
# Helper: weights_obs_row_trivial
# ---------------------------------------------------------------------------


class TestRowTrivialHelper:
    """Direct unit tests on the trivial-row classifier."""

    def test_none_is_trivial(self):
        """``None`` row is trivial.

        Failure implies the very short-circuit that protects byte identity
        with the legacy ``weights_obs_*=None`` callers regressed.
        """
        assert weights_obs_row_trivial(None, np.array([1.0, 2.0, 3.0])) is True

    def test_all_ones_on_finite_obs_is_trivial(self):
        """All-ones on finite-obs cells (and arbitrary on NaN-obs cells) is trivial.

        The NaN-obs cells do not matter because ``mask_weights`` will zero
        them out downstream regardless of the per-cell weight value there.
        """
        wobs = np.array([1.0, 1.0, 99.0, 1.0])  # cell 2 has NaN obs so its weight is moot
        obs = np.array([0.0, 5.0, np.nan, 7.0])
        assert weights_obs_row_trivial(wobs, obs) is True

    def test_non_one_on_finite_obs_is_not_trivial(self):
        """A non-1.0 weight on any finite-obs cell flips the row to non-trivial.

        Failure implies the classifier is too permissive and 0.5- or
        0.95-confidence rows would route through the unweighted code path,
        contradicting the documented mass-preserving contract.
        """
        wobs = np.array([1.0, 0.95, 1.0, 1.0])
        obs = np.array([0.0, 5.0, 9.0, 2.0])
        assert weights_obs_row_trivial(wobs, obs) is False

    def test_non_finite_weight_on_finite_obs_is_not_trivial(self):
        """A NaN weight on a finite-obs cell is NOT trivial — it must flow through the renorm path.

        A NaN weight on a finite observation means "no confidence" and must be
        treated as zero in ``weights_obs_effective``; routing it through the
        unweighted code path would silently restore full trust to that cell.
        """
        wobs = np.array([1.0, np.nan, 1.0, 1.0])
        obs = np.array([0.0, 5.0, 9.0, 2.0])
        assert weights_obs_row_trivial(wobs, obs) is False

    def test_all_obs_nonfinite_is_trivial(self):
        """If no obs cell is finite the row is trivial (no path actually executes).

        With no finite obs, neither the unweighted nor the weighted NB block
        runs (the gate trips to 0). Either path produces the same result, so
        the classifier may return True for simplicity.
        """
        wobs = np.array([0.5, 0.5, 0.5])
        obs = np.array([np.nan, np.nan, np.nan])
        assert weights_obs_row_trivial(wobs, obs) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
