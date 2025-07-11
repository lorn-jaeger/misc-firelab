import pandas as pd
import geopandas as gpd
from shapely.geometry import box

import pandas as pd
from pathlib import Path

data_dir = Path("./air_quality/data/final")
csv_files = list(data_dir.glob("*.csv"))
data = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)

data["Time"] = pd.to_datetime(data["Time"])
data["MERRA2R_trimmed"] = data.loc[data["Time"].dt.year != 2018, "MERRA2R"]

fires = pd.read_pickle("air_quality/data/fires.pkl")

data["Time"] = pd.to_datetime(data["Time"])
fires["IDate"] = pd.to_datetime(fires["IDate"])
fires["FDate"] = pd.to_datetime(fires["FDate"])

fires["geometry"] = fires.apply(
    lambda row: box(row["lon"] - 1, row["lat"] - 1, row["lon"] + 1, row["lat"] + 1),
    axis=1
)
fires_gdf = gpd.GeoDataFrame(fires, geometry="geometry", crs="EPSG:4326")

min_date = fires["IDate"].min()
max_date = fires["FDate"].max()

data_subset = data[(data["Time"] >= min_date) & (data["Time"] <= max_date)]

data_subset["geometry"] = gpd.points_from_xy(data_subset["Longitude"], data_subset["Latitude"])
data_gdf = gpd.GeoDataFrame(data_subset, geometry="geometry", crs="EPSG:4326")

joined = gpd.sjoin(data_gdf, fires_gdf, predicate="within", how="inner")

filtered = joined[(joined["Time"] >= joined["IDate"]) & (joined["Time"] <= joined["FDate"])]

fires = filtered[data.columns]

fires.to_csv("fires2.csv")

