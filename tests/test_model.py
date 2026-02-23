import tempfile
import unittest
from pathlib import Path

import pytest

from laser_cholera.metapop.model import run_model
from laser_cholera.metapop.params import get_parameters
from laser_cholera.utils import sim_duration


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


if __name__ == "__main__":
    unittest.main()
