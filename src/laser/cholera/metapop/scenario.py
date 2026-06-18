"""Static MOSAIC-country scenario: shapefile geometry joined to 2023 demographics.

Loads the bundled `mosaic_countries.shp` (country polygons) and
`demographics_africa_2000_2023.csv` (per-year per-country populations),
filters the population panel to year 2023 and the MOSAIC ISO subset,
then inner-joins on ISO code. The result is exposed as the module-level
`scenario` `GeoDataFrame` for downstream plotting / scenario authoring.

Side effect: `make_scenario()` runs at import time, so the I/O cost
(shapefile parse + CSV read) is paid once.
"""

import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geopandas as gpd
import pandas as pd

from laser.cholera import iso_codes

__all__ = ["scenario"]


def make_scenario() -> pd.DataFrame:
    """Build the MOSAIC-country geometry-plus-demographics joined frame.

    Reads `data/mosaic_countries.shp` and the 2023 row of
    `data/demographics_africa_2000_2023.csv`, restricts to ISO codes in
    [`laser.cholera.iso_codes`][laser.cholera.iso_codes], and inner-joins
    on ISO.

    Returns:
        A `GeoDataFrame` (returned as `pd.DataFrame` for typing
        simplicity) with one row per MOSAIC country, holding both the
        shapefile geometry and the 2023 population columns.
    """
    shape_data = gpd.read_file(Path(__file__).parent.absolute() / "data" / "mosaic_countries.shp")
    populations = pd.read_csv(Path(__file__).parent.absolute() / "data" / "demographics_africa_2000_2023.csv")
    twenty_twentythree = populations[populations.year == 2023]
    mosaic_populations = twenty_twentythree[twenty_twentythree.iso_code.isin(iso_codes)]
    merged = pd.merge(shape_data, mosaic_populations, left_on=["ISO"], right_on=["iso_code"])

    return merged


scenario = make_scenario()
