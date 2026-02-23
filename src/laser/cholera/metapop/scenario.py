import warnings
from pathlib import Path

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geopandas as gpd
import pandas as pd

from laser_cholera import iso_codes

__all__ = ["scenario"]


def make_scenario() -> pd.DataFrame:
    shape_data = gpd.read_file(Path(__file__).parent.absolute() / "data" / "mosaic_countries.shp")
    populations = pd.read_csv(Path(__file__).parent.absolute() / "data" / "demographics_africa_2000_2023.csv")
    twenty_twentythree = populations[populations.year == 2023]
    mosaic_populations = twenty_twentythree[twenty_twentythree.iso_code.isin(iso_codes)]
    merged = pd.merge(shape_data, mosaic_populations, left_on=["ISO"], right_on=["iso_code"])

    return merged


scenario = make_scenario()
