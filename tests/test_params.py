from pathlib import Path
from unittest import TestCase

from laser_cholera.metapop.params import get_parameters

SCRIPT_DIR = Path(__file__).parent.absolute()
PARAMS_DIR = SCRIPT_DIR / "../src/laser_cholera/metapop/data"


class TestGetParameters(TestCase):
    def test_load_uncompressed_json(self):
        _params = get_parameters(PARAMS_DIR / "default_parameters.json", mods={"loglevel": "DEBUG"})
        # assert True

        return

    def test_load_compressed_json(self):
        _params = get_parameters(PARAMS_DIR / "default_parameters.json.gz", mods={"loglevel": "DEBUG"})
        # assert True

        return

    # def test_load_uncompressed_hdf5(self):
    #     _params = get_parameters(PARAMS_DIR / "default_parameters.h5", overrides={"loglevel": "DEBUG"})
    #     # assert True

    #     return

    # def test_load_compressed_hdf5(self):
    #     _params = get_parameters(PARAMS_DIR / "default_parameters.h5.gz", overrides={"loglevel": "DEBUG"})
    #     # assert True

    #     return


if __name__ == "__main__":
    from unittest import main

    main()
