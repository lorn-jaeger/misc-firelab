from pathlib import Path
import glob
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import argparse
import warnings
from ee.geometry import Geometry
from ee.imagecollection import ImageCollection
from tqdm import tqdm
from pyproj import Proj, Transformer
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geemap

SENSOR_PATH = Path("data/sensors/using")
TEST_PATH = Path("data/test/sensors")
OUT_PATH = Path("data/out")

def clear_output():
    """
    Clear the csv output.
    """
    for file in OUT_PATH.iterdir():
        if file.is_file():
            file.unlink()


def parse_args():
    """
    Simple argument parser with some useful commands.
    Allows me to set a test dir and clear the output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-out", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.clear_out:
        clear_output() 

    global SENSOR_PATH

    if args.test:
        SENSOR_PATH = TEST_PATH

def save(file, name):
    file.to_csv(f"data/out/out_{name}", index=False)


import re

def get_year(name: str) -> int:
    match = re.search(r"\d{4}", name)
    if match:
        return int(match.group(0))
    raise ValueError(f"No 4-digit year found in filename: {name}")

def try_auth():
    geemap.ee_initialize()


def fmt_sensors(df):
    date = pd.to_datetime(df["Date GMT"])           # UTC daylight-agnostic
    time = pd.to_timedelta(df["Time GMT"], unit="h")
    df["Time"] = (date + time).dt.tz_localize("UTC")
    df["Time_cst"] = df["Time"].dt.tz_convert("Etc/GMT+6").tz_localize(None)
    return df[["Time_utc", "Time_cst", "Latitude", "Longitude",
               "Sample Measurement"]].assign(CONUS=pd.NA, MERRA2=pd.NA,
                                             MERRA2R=pd.NA, CAMS=pd.NA)

def get_unique(sensors):
    result = sensors.groupby(['Latitude', 'Longitude'])['Time'].agg(['min', 'max']).reset_index()
    result.rename(columns={'min': 'Start Time', 'max': 'End Time'}, inplace=True)

    return result



def fmt_ee_output(file):
    pass

def CAMS(sensors, name):
    sensors["CAMS"] = pd.NA

    data = pd.read_csv(f"./data/ee/cams_{name}", low_memory=False)

    data = data.rename(columns={
        "lat": "Latitude",
        "lon": "Longitude",
        "time": "Time",
    })

    sensors["Time"] = pd.to_datetime(sensors["Time"])
    data["Time"] = pd.to_datetime(data["Time"])

    data = data[["Latitude", "Longitude", "Time", "first"]]

    sensors = sensors.merge(data, on=["Latitude", "Longitude", "Time"], how="left")
    sensors["CAMS"] = sensors["first"]
    sensors["CAMS"] = sensors["CAMS"] * 1_000_000_000
    sensors = sensors.drop(columns=["first"])

    return sensors


def MERRA2(sensors, name):
    sensors = sensors.copy()
    year = name.split("_")[-1].split(".")[0]
    nc4_dir = Path("./data/merra2/")
    nc4_files = sorted(nc4_dir.glob(f"*Nx.{year}*.nc4"))

    ds = xr.open_mfdataset(nc4_files, combine="by_coords")

    sensors["time_np"] = pd.to_datetime(sensors["Time"]).astype("datetime64[ns]")
    sensors["lat"] = sensors["Latitude"]
    sensors["lon"] = sensors["Longitude"]

    variables = [
        "SSSMASS25",   
        "OCSMASS",    
        "BCSMASS",   
        "SO4SMASS",  
        "DUSMASS25" 
    ]

    for var in variables:
        interp = ds[var].interp(
            time=xr.DataArray(sensors["time_np"], dims="sensor"),
            lat=xr.DataArray(sensors["lat"], dims="sensor"),
            lon=xr.DataArray(sensors["lon"], dims="sensor"),
            method="nearest"
        )
        sensors[var] = interp.values

    '''
    HOW DO I OBTAIN SURFACE PM2.5 CONCENTRATION IN MERRA-2?

    Using fields from the 2D tavg1_2d_aer_Nx collection, the concentration of particulate matter can be computed using the following formula: PM2.5 = DUSMASS25 + OCSMASS+ BCSMASS + SSSMASS25 + SO4SMASS* (132.14/96.06) 
    '''

    sensors["MERRA2"] = (
            sensors["DUSMASS25"]
            + sensors["OCSMASS"]
            + sensors["BCSMASS"]
            + sensors["SSSMASS25"]
            + sensors["SO4SMASS"] * (132.14 / 96.06)
    )  * 1_000_000_000

    sensors = sensors[["Time", "Latitude", "Longitude", "Sample Measurement", "MERRA2R", "MERRA2", "CONUS", "CAMS"]]

    return sensors

def MERRA2R(sensors, name):
    sensors = sensors.copy()
    sensors["MERRA2R"] = pd.NA
    year = name.split("_")[-1].split(".")[0]
    nc4_dir = Path("./data/merra2r/")
    nc4_files = sorted(nc4_dir.glob(f"*V1.{year}*.nc4*"))

    ds = xr.open_mfdataset(nc4_files, combine="by_coords", decode_times=True)

    sensors["time_np"] = pd.to_datetime(sensors["Time"]).astype("datetime64[ns]")
    sensors["lat"] = sensors["Latitude"]
    sensors["lon"] = sensors["Longitude"]

    var = "MERRA2_CNN_Surface_PM25" 

    interp = ds[var].interp(
        time=xr.DataArray(sensors["time_np"], dims="sensor"),
        lat=xr.DataArray(sensors["lat"], dims="sensor"),
        lon=xr.DataArray(sensors["lon"], dims="sensor"),
        method="nearest"
    ) 

    sensors["MERRA2R"] = interp.values

    sensors = sensors[["Time", "Latitude", "Longitude", "Sample Measurement", "MERRA2R", "MERRA2", "CONUS", "CAMS"]]

    return sensors


def CONUS(sensors, name):
    sensors = sensors.copy()
    years = sensors["Time"].dt.year.unique()

    directory = Path("./data/conus")
    files = sorted(str(f) for f in directory.glob(f"*{years[0]}*.nc"))

    ds = xr.open_mfdataset(
        files,
        combine="nested",
        concat_dim="TSTEP",
        decode_cf=False
    )

    sdate = ds.attrs["SDATE"]  
    tstep = ds.attrs["TSTEP"] 

    year = sdate // 1000
    doy = sdate % 1000

    start_time = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)

    nt = ds.sizes["TSTEP"]
    datetimes = [start_time + timedelta(hours=i) for i in range(nt)]

    ds = ds.assign_coords(time=("TSTEP", datetimes))

    proj = Proj(
        proj="lcc",
        lat_1=ds.attrs["P_ALP"],
        lat_2=ds.attrs["P_BET"],
        lat_0=ds.attrs["YCENT"],
        lon_0=ds.attrs["XCENT"],
        x_0=0,
        y_0=0,
    )

    transformer = Transformer.from_proj("epsg:4326", proj, always_xy=True)

    lon_arr = sensors["Longitude"].to_numpy()
    lat_arr = sensors["Latitude"].to_numpy()
    x_arr, y_arr = transformer.transform(lon_arr, lat_arr)

    XORIG, YORIG = ds.attrs["XORIG"], ds.attrs["YORIG"]
    XCELL, YCELL = ds.attrs["XCELL"], ds.attrs["YCELL"]


    col = np.round((x_arr - XORIG) / XCELL).astype("int64")
    row = np.round((y_arr - YORIG) / YCELL).astype("int64")

    in_bounds = (
        (0 <= col) & (col < ds.dims["COL"]) &
        (0 <= row) & (row < ds.dims["ROW"])
    )

    time_idx = ds.indexes["time"].get_indexer(sensors["Time_cst"].to_numpy(), method="nearest")
    time_ok = time_idx >= 0

    valid = in_bounds & time_ok

    pm25_grid = ds["PM25_TOT"].isel(LAY=0).values * 1_000_000_000

    out = pd.Series(pd.NA, index=sensors.index, name="CONUS", dtype="Float64")
    if valid.any():
        vals = pm25_grid[
            time_idx[valid],
            row[valid],
            col[valid]
        ]
        out.iloc[valid] = vals

    sensors["CONUS"] = out
    return sensors


def main() -> None:
    parse_args()
    try_auth()

    files = SENSOR_PATH.iterdir()
    sources = [CONUS]

    for file in files:
        print(f"Reading {file.name}")
        sensors = pd.read_csv(file, low_memory=False)
        sensors = fmt_sensors(sensors)
        print(f"Processing {file.name}")
        for source in sources:
            print(f"{source.__name__}...")
            sensors = source(sensors, file.name)
            print(sensors)
        save(sensors, file.name)

if __name__ == "__main__":
    main()


# def parse_gee_region(raw):
#     headers = raw[0]
#     data = raw[1:]
#
#     df = pd.DataFrame(data, columns=headers)
#
#     df = df[df["time"].apply(lambda x: isinstance(x, (int, float)))]
#
#     df["Time"] = pd.to_datetime(df["time"], unit="ms")
#     df["Longitude"] = df["longitude"].astype(float)
#     df["Latitude"] = df["latitude"].astype(float)
#     df["pm25"] = df["particulate_matter_d_less_than_25_um_surface"].astype(float)
#
#     df["pm25"] *= 1_000_000_000
#
#     return df[["Time", "Longitude", "Latitude", "pm25"]]
#
# def CAMS(sensors):
#     sensors["CAMS"] = pd.NA
#
#     cams = (
#         ImageCollection("ECMWF/CAMS/NRT")
#         .select("particulate_matter_d_less_than_25_um_surface")
#         .filter('model_initialization_hour == 0')
#     )
#
#     cams_data = pd.DataFrame(columns=["Time", "Longitude", "Latitude", "pm25"]) #type: ignore
#
#     locations = get_unique(sensors)
#     outer = tqdm(locations.iterrows(), total=len(locations), desc="Fetching CAMS data", position=0)
#     # can use apply instead but this is only 1000 iterations
#     # most of the time is taken by api calls to earth engine
#     for _, row in outer:
#         point = Geometry.Point(row["Longitude"], row["Latitude"])
#         # batching is needed so we don't hit the 3000 item limit
#         months = pd.date_range(pd.to_datetime(row["Start Time"]), pd.to_datetime(row["End Time"]), freq="ME")
#         inner = tqdm(months, desc="Monthly data", leave=False, position=1)
#         for month in inner:
#             start = month
#             end = month + pd.DateOffset(months=1)
#
#             # skip if start is before the start of the dataset
#             if start < pd.Timestamp("2016-06-23"):
#                 continue
#
#             try:
#
#                 monthly_data = (
#                     cams
#                     .filterBounds(point)
#                     .filterDate(start, end)
#                     .getRegion(point, scale=10_000)
#                     .getInfo()
#                 )
#                 monthly_data = parse_gee_region(monthly_data)
#                 # Needed overwrite. Google Earth Engine rounds lats and lons so there are slightly off
#                 # Otherwise there will be no matches on merge
#                 monthly_data["Latitude"] = row["Latitude"]
#                 monthly_data["Longitude"] = row["Longitude"]
#                 cams_data = pd.concat([cams_data, monthly_data], ignore_index=True) #type: ignore
#             except Exception as e:
#                 print(e)
#                 continue
#
#
#     sensors = sensors.merge(
#         cams_data,
#         on=["Time", "Latitude", "Longitude"],
#         how="left"
#     )
#
#     sensors["CAMS"] = sensors["pm25"]
#     sensors.drop(columns=["pm25"], inplace=True)
#
#     return sensors
#


# from earthaccess import login, search_data, download
# import xarray as xr
# import pandas as pd
# from tqdm import tqdm
# from ee.geometry import Geometry
#
#
# from earthaccess import search_data, download
# import xarray as xr
# import pandas as pd
# from tqdm import tqdm
# from pathlib import Path
#
# def MERRA2(sensors):
#     sensors["MERRA2"] = pd.NA
#     variable = "MERRA2_CNN_Surface_PM25"
#     data = pd.DataFrame(columns=["Time", "Latitude", "Longitude", "pm25"])
#
#     locations = get_unique(sensors)
#
#     for _, row in locations.iterrows():
#         lat, lon = row["Latitude"], row["Longitude"]
#         start = pd.to_datetime(row["Start Time"])
#         end = pd.to_datetime(row["End Time"])
#         print(f"Processing sensor at {lat}, {lon} from {start} to {end}")
#
#         results = search_data(
#             concept_id="C3094710982-GES_DISC",
#             temporal=(start, end),
#             bounding_box=(lon - 0.1, lat - 0.1, lon + 0.1, lat + 0.1),
#         )
#         files = download(results)
#         if not files:
#             continue
#
#         try:
#             ds = xr.open_mfdataset(files, combine="by_coords")
#             times = pd.date_range(start, end, freq="H")
#             sat_pm25 = ds[variable].interp(
#                 time=xr.DataArray(times, dims="time"),
#                 lat=lat,
#                 lon=lon
#             )
#             df = pd.DataFrame({
#                 "Time": times,
#                 "Latitude": lat,
#                 "Longitude": lon,
#                 "pm25": sat_pm25.values
#             })
#             data = pd.concat([data, df], ignore_index=True)
#         except Exception as e:
#             print(f"Failed {lat},{lon}: {e}")
#         finally:
#             ds.close()
#             for f in files:
#                 try:
#                     Path(f).unlink()
#                 except Exception as e:
#                     print(f"Error deleting file {f}: {e}")
#
#     # Round to avoid floating-point merge mismatch
#     data["Latitude"] = data["Latitude"].round(4)
#     data["Longitude"] = data["Longitude"].round(4)
#     sensors["Latitude"] = sensors["Latitude"].round(4)
#     sensors["Longitude"] = sensors["Longitude"].round(4)
#
#     sensors = sensors.merge(data, on=["Time", "Latitude", "Longitude"], how="left")
#     sensors["MERRA2"] = sensors["pm25"]
#     sensors.drop(columns=["pm25"], inplace=True)
#
#     print(sensors)
#     print(sensors.shape)
#
#     return sensors
#
#

