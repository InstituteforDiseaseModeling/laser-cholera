import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest

from laser.cholera.metapop.model import run_model
from laser.cholera.metapop.params import get_parameters
from laser.cholera.utils import sim_duration


class TestModel(unittest.TestCase):
    @staticmethod
    def get_test_parameters(overrides=None, trim=True):
        params = get_parameters(mods=overrides if overrides else sim_duration(), do_validation=False)

        if trim:
            # Trim the parameters to test duration for testing
            params.b_jt = params.b_jt[: params.nticks, :]
            params.d_jt = params.d_jt[: params.nticks, :]
            params.nu_1_jt = params.nu_1_jt[: params.nticks, :]
            params.nu_2_jt = params.nu_2_jt[: params.nticks, :]
            params.psi_jt = params.psi_jt[: params.nticks, :]

        # S - use given susceptible populations
        # E - move any exposed people back to susceptible
        params.S_j_initial += params.E_j_initial
        params.E_j_initial[:] = 0
        # I - move any infectious people back to susceptible
        params.S_j_initial += params.I_j_initial
        params.I_j_initial[:] = 0
        # R - move any recovered people back to susceptible
        params.S_j_initial += params.R_j_initial
        params.R_j_initial[:] = 0
        # V1 and V2 - move any vaccinated people back to susceptible
        params.S_j_initial += params.V1_j_initial + params.V2_j_initial
        params.V1_j_initial[:] = 0
        params.V2_j_initial[:] = 0

        return params

    def test_run_model_None(self):
        # Test calling run_model with None for parameters (use defaults)
        run_model(None)

        assert True, "run_model with None parameters should not raise an error."

        return

    def test_run_model_string(self):
        parameters = self.get_test_parameters(None)  # Get the default parameters

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
            parameters.save(tmpfile.name)
            try:
                run_model(tmpfile.name)
            except Exception as e:
                self.fail(f"run_model with string parameter filename raised an error: {e}")

        assert True, "run_model with string parameter filename should not raise an error."

        return

    def test_run_model_path(self):
        parameters = self.get_test_parameters(None)  # Get the default parameters

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmpfile:
            parameters.save(tmpfile.name)
            try:
                run_model(Path(tmpfile.name))
            except Exception as e:
                self.fail(f"run_model with Path parameter filename raised an error: {e}")

        assert True, "run_model with Path parameter file should not raise an error."

        return

    def test_run_model_dict(self):
        parameters = self.get_test_parameters(None)  # Get the default parameters
        parameters_dict = parameters.to_dict()

        run_model(parameters_dict)

        assert True, "run_model with dict of parameters should not raise an error."

        return

    def test_run_model_invalid(self):
        with pytest.raises(ValueError, match="Invalid parameter source type"):
            run_model(3.14159265)

        return

    def _assert_channel_buffer_and_results_view(self, channel, patches_buf, results_view, nticks, nnodes):
        """Helper — assert the four buffer + R-view invariants for one per-tick channel.

        For a channel that is allocated by its component with
        `length=nticks + 1` and written by the per-tick distribution at indices
        `tick` for `tick in range(nticks)`, the buffer on `model.patches` has
        shape `(nticks + 1, nnodes)` with the LAST row left as an unwritten
        all-zeros sentinel; the matching view on `model.results` is the FIRST
        `nticks` rows of the buffer transposed for R-side consumers, shape
        `(nnodes, nticks)`.

        Asserts:

        1. `patches_buf.shape == (nticks + 1, nnodes)`.
        2. `patches_buf[-1, :]` is all zeros (the sentinel).
        3. `results_view.shape == (nnodes, nticks)`.
        4. `results_view` is bit-equal to `patches_buf[:-1, :].T` — the FIRST
           `nticks` rows of the buffer, transposed.
        5. (Conditional) When the buffer's trajectory is non-uniform across
           rows (`patches_buf[:-1, :] != patches_buf[1:, :]`), the view is NOT
           equal to `patches_buf[1:, :].T` (the LAST `nticks` rows transposed),
           which would be the symptom of a regression that dropped the FIRST
           row of the buffer instead of the LAST. When the buffer trajectory
           is uniform (e.g., the channel saw no per-tick activity over the
           short test window), the swap-detection assertion is vacuous and
           silently skipped — the shape / sentinel / content checks above
           still ran.

        Args:
            channel: channel name (used only in failure messages).
            patches_buf: `model.patches.<channel>` ndarray.
            results_view: `model.results.<channel>` ndarray.
            nticks: `params.nticks`.
            nnodes: `len(params.location_name)`.
        """
        # (1) Patches buffer shape.
        assert patches_buf.shape == (nticks + 1, nnodes), (
            f"model.patches.{channel} shape mismatch: expected ({nticks + 1}, {nnodes}), got {patches_buf.shape}"
        )
        # (2) Sentinel zeros at the last row.
        np.testing.assert_array_equal(
            patches_buf[-1, :],
            np.zeros(nnodes, dtype=patches_buf.dtype),
            err_msg=(
                f"model.patches.{channel}[-1, :] is not all zeros — the sentinel row got written into; downstream views will be off by one tick"
            ),
        )
        # (3) Results view shape: R-style transposed.
        assert results_view.shape == (nnodes, nticks), (
            f"model.results.{channel} shape mismatch: expected ({nnodes}, {nticks}), got {results_view.shape}"
        )
        # (4) Results view == FIRST nticks rows of patches, transposed.
        np.testing.assert_array_equal(
            results_view,
            patches_buf[:-1, :].T,
            err_msg=(f"model.results.{channel} drifted from model.patches.{channel}[:-1, :].T (first-nticks rows of the buffer, transposed)"),
        )
        # (5) Results view != LAST nticks rows transposed (the swapped slice) —
        # only meaningful when the trajectory is non-uniform across rows.
        if not np.array_equal(patches_buf[:-1, :], patches_buf[1:, :]):
            assert not np.array_equal(results_view, patches_buf[1:, :].T), (
                f"model.results.{channel} matches the LAST-nticks-rows transposed slice "
                f"(patches.{channel}[1:, :].T) — the R-interface may have silently switched "
                "to dropping the FIRST row of the buffer instead of the LAST."
            )

    def test_per_tick_channels_buffer_and_results_view(self):
        """Verify `model.patches.<channel>` buffer layout and `model.results.<channel>` R-style view for five per-tick channels.

        Channels covered: `reported_cases`, `reported_deaths`, `births`,
        `disease_deaths`, `non_disease_deaths`. Each is allocated by its
        owning component (`Infectious` for `reported_*` and `disease_deaths`;
        `Susceptible` / `Exposed` / `Recovered` / `Vaccinated` for `births`
        and `non_disease_deaths`) with `length=nticks + 1`, and its per-tick
        distribution writes to `patches.<channel>[tick]` for `tick in
        range(nticks)`. Indices `0..nticks-1` therefore carry the per-tick
        results and index `nticks` (the last row) is the never-written
        all-zeros sentinel.

        `RInterface` then exposes the FIRST `nticks` rows of each buffer,
        transposed for R-side consumers, on `model.results`. The view has
        shape `(nnodes, nticks)` and content `patches_buf[:-1, :].T`.

        Given the default initial infections kept in place (so each channel
        accumulates a non-trivial trajectory) and `rho` / `rho_deaths` pinned
        to `1.0`, when the model finishes, then for EACH of the five
        channels (asserted independently via `self.subTest(channel=…)`):

        1. `model.patches.<channel>.shape == (nticks + 1, nnodes)`.
        2. `model.patches.<channel>[-1, :]` is all zeros — the sentinel row
           that no per-tick distribution ever writes into.
        3. `model.results.<channel>.shape == (nnodes, nticks)`.
        4. `model.results.<channel>` is bit-equal to
           `model.patches.<channel>[:-1, :].T` — the FIRST `nticks` rows
           transposed. NOT equal to `model.patches.<channel>[1:, :].T`
           (the LAST `nticks` rows transposed); the swap-detection sub-check
           runs only when the buffer trajectory is non-uniform.

        Failure on any channel surfaces a buffer-layout or R-view drift that
        downstream consumers (analyzer plots, `calc_model_likelihood`,
        R-side notebooks) would silently mis-align by one tick.
        """
        overrides = sim_duration()  # default 2025-03-24 → 2025-04-24, nticks=32
        overrides["rho"] = 1.0
        overrides["rho_deaths"] = 1.0

        params = get_parameters(mods=overrides, do_validation=False)

        # Trim per-tick parameter arrays to the shortened nticks. The existing
        # `get_test_parameters` helper on this class also moves every non-S
        # compartment back to S, which would zero out the initial infections and
        # produce all-zero per-tick channels — defeating the swap-detection
        # check. So we trim manually here and keep the defaults' initial
        # Isym / Iasym.
        for attr in ("b_jt", "d_jt", "nu_1_jt", "nu_2_jt", "psi_jt"):
            setattr(params, attr, getattr(params, attr)[: params.nticks, :])

        model = run_model(params.to_dict())

        nticks = params.nticks
        nnodes = len(params.location_name)

        for channel in (
            "reported_cases",
            "reported_deaths",
            "births",
            "disease_deaths",
            "non_disease_deaths",
        ):
            with self.subTest(channel=channel):
                patches_buf = getattr(model.patches, channel)
                results_view = getattr(model.results, channel)
                self._assert_channel_buffer_and_results_view(channel, patches_buf, results_view, nticks, nnodes)

        return


if __name__ == "__main__":
    unittest.main()
