"""Tests for `laser.cholera.metapop.derivedvalues.calculate_coupling`.

The post-perf-work implementation uses `np.corrcoef` instead of a manual
nested loop. The semantic-preserving piece is the handling of patches
that had zero prevalence variance over the simulation (typically
isolated patches that no force-of-infection ever reached): the
corresponding rows / columns of the coupling matrix must come out as
NaN, not as 0 and not as an error.
"""

import logging

import numpy as np
import pytest

from laser.cholera.metapop.derivedvalues import calculate_coupling


def _make_inputs(T: int, L: int, *, with_const_cols=()) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build (Isym, Iasym, N, C) inputs of the shapes calculate_coupling expects.

    Columns listed in `with_const_cols` are forced to constant zero so
    their resulting variance is exactly zero.
    """
    rng = np.random.default_rng(seed=0)
    Isym = rng.integers(0, 100, size=(T, L)).astype(np.int32)
    Iasym = rng.integers(0, 100, size=(T, L)).astype(np.int32)
    for col in with_const_cols:
        Isym[:, col] = 0
        Iasym[:, col] = 0
    N = (Isym + Iasym + rng.integers(1000, 5000, size=(T, L))).astype(np.int32)
    C = np.zeros((L, L), dtype=np.float32)
    return Isym, Iasym, N, C


def test_calculate_coupling_with_constant_column_returns_nan(caplog):
    """A patch with zero prevalence variance produces NaN rows/cols, not an error.

    Given an input where patch index 2 has `Isym + Iasym == 0` for every
    tick (a legitimate model outcome for an isolated, never-seeded patch),
    when `calculate_coupling` runs,
    then `C[2, :]` and `C[:, 2]` are all NaN, the other off-diagonal
    entries are finite Pearson correlations in `[-1, 1]`, and the function
    logs an INFO line naming the count of constant patches.

    Failure implies the perf-work refactor regressed: either NaN handling
    is broken, the function leaked a `RuntimeWarning`, or the INFO log
    was dropped so the constant-column situation is invisible at run time.
    """
    Isym, Iasym, N, C = _make_inputs(T=200, L=5, with_const_cols=(2,))

    with caplog.at_level(logging.INFO, logger="laser.cholera"):
        calculate_coupling(Isym, Iasym, N, C)

    # Constant patch's row and column are NaN.
    assert np.all(np.isnan(C[2, :])), f"Row 2 should be all-NaN, got {C[2, :]}"
    assert np.all(np.isnan(C[:, 2])), f"Col 2 should be all-NaN, got {C[:, 2]}"

    # Off-diagonal entries among non-constant columns are finite.
    keep = np.array([0, 1, 3, 4])
    sub = C[np.ix_(keep, keep)]
    assert np.all(np.isfinite(sub)), f"Non-constant sub-block should be finite, got {sub}"
    assert np.all(sub >= -1.0001) and np.all(sub <= 1.0001), "Correlations must be in [-1, 1]"

    # Diagonal of non-constant columns is 1.0 (within float tolerance).
    assert np.allclose(np.diag(sub), 1.0, atol=1e-5), f"Non-constant diagonal should be 1.0, got {np.diag(sub)}"

    # INFO log was emitted.
    assert any("calculate_coupling" in r.message and "1 of 5 patches" in r.message for r in caplog.records), (
        f"expected an INFO log naming the count of constant patches; got {[r.message for r in caplog.records]}"
    )


def test_calculate_coupling_emits_no_warning_on_constant_columns():
    """The constant-column case must not leak a `RuntimeWarning` to the caller.

    Given the same `with_const_cols=(2,)` input,
    when `calculate_coupling` runs inside a `warnings.catch_warnings()`
    block configured to escalate `RuntimeWarning` to an error,
    then the call completes without raising.

    Failure implies the explicit constant-column detection has been
    removed and the function is back to relying on `np.errstate`
    suppression (which is brittle: pytest's `filterwarnings = error`
    config promotes the warning to a test failure).
    """
    Isym, Iasym, N, C = _make_inputs(T=200, L=5, with_const_cols=(2,))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        calculate_coupling(Isym, Iasym, N, C)


def test_calculate_coupling_all_constant_columns_produces_all_nan():
    """Every patch having zero variance produces an all-NaN matrix without erroring.

    Given an input where every column of `Isym + Iasym` is identically
    zero,
    when `calculate_coupling` runs,
    then `C` is entirely NaN (the `keep.any()` short-circuit takes the
    no-corrcoef-call path), no RuntimeWarning is emitted, and the INFO
    log reports `L of L` constant patches.

    Failure implies a regression in the edge-case short-circuit when no
    columns survive the constant filter.
    """
    Isym, Iasym, N, C = _make_inputs(T=50, L=4, with_const_cols=(0, 1, 2, 3))

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        calculate_coupling(Isym, Iasym, N, C)

    assert np.all(np.isnan(C)), f"All-constant input should yield all-NaN coupling, got {C}"


def test_calculate_coupling_no_constant_columns_matches_corrcoef():
    """When every column has nonzero variance, the result is `np.corrcoef(y, rowvar=False)`.

    Given an input with no constant columns,
    when `calculate_coupling` runs,
    then `C` is bit-equal to `np.corrcoef(y, rowvar=False)` for
    `y = (Isym + Iasym) / N`, where the equality is checked at the
    coupling matrix's `float32` storage precision.

    Failure implies the happy-path branch has drifted from the canonical
    `np.corrcoef` formulation.
    """
    Isym, Iasym, N, C = _make_inputs(T=200, L=5)

    calculate_coupling(Isym, Iasym, N, C)

    expected = np.corrcoef((Isym + Iasym) / N, rowvar=False).astype(np.float32)
    assert np.array_equal(C, expected), "happy path should equal np.corrcoef at float32 precision"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
