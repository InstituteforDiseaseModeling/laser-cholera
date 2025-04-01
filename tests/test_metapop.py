import unittest

import click

from laser_cholera.metapop.model import cli_run


class TestCholeraMPM(unittest.TestCase):
    def test_cholera_metapopulation_model(self):
        # Run the metapopulation model with default parameters
        ctx = click.Context(cli_run)
        ctx.invoke(cli_run, seed=20250326, verbose=False, no_viz=True, pdf=False)


if __name__ == "__main__":
    unittest.main()
