from pathlib import Path
import glob
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

SENSOR_PATH = Path("data/sensors")
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
    file.to_csv(f"data/out/out{name}", index=False)

def try_auth():
    geemap.ee_initialize()

def fmt_sensors(sensors):
    sensors['Date GMT'] = pd.to_datetime(sensors['Date GMT'])
    sensors['Time'] = sensors['Date GMT'] + pd.to_timedelta(sensors['Time GMT'] + ':00')
    
    sensors = sensors[["Time", "Latitude", "Longitude", "Sample Measurement"]]

    return sensors

def get_unique(sensors):
    result = sensors.groupby(['Latitude', 'Longitude'])['Time'].agg(['min', 'max']).reset_index()
    result.rename(columns={'min': 'Start Time', 'max': 'End Time'}, inplace=True)

    return result

def AIRNOW(sensors, name):
    sensors["AIRNOW"] = pd.NA
    return sensors

def MERRA2(sensors, name):
    sensors["MEERA2R"] = pd.NA

    return sensors

def MERRA2R(sensors, name):
    sensors["MEERA2R"] = pd.NA

    return sensors

def decode_times(ds):
    sdate = ds.attrs["SDATE"]
    tstep = ds.attrs["TSTEP"]

    year = int(str(sdate)[:4])
    jday = int(str(sdate)[4:])
    base_time = datetime(year, 1, 1) + timedelta(days=jday - 1)

    step_hours = int(tstep // 10000)
    times = [base_time + timedelta(hours=int(i) * step_hours) for i in range(ds.dims["TSTEP"])]

    ds = ds.assign_coords(time=("TSTEP", times))
    ds = ds.swap_dims({"TSTEP": "time"})  
    return ds

def get_pm25_CONUS(row, ds, transformer):
    x, y = transformer.transform(row["Longitude"], row["Latitude"])

    XORIG = ds.attrs['XORIG']
    YORIG = ds.attrs['YORIG']
    XCELL = ds.attrs['XCELL']
    YCELL = ds.attrs['YCELL']

    col = int((x - XORIG) / XCELL)
    row_idx = int((y - YORIG) / YCELL)

    if not (0 <= col < ds.dims["COL"]) or not (0 <= row_idx < ds.dims["ROW"]):
        return pd.NA

    try:
        time_idx = ds.indexes["time"].get_indexer([row["Time"]], method="nearest")[0]
        pm25 = ds['PM25_TOT'].isel(time=time_idx, LAY=0, ROW=row_idx, COL=col).values.item()
    except Exception as e:
        print(f"Failed for time: {row['Time']}, row: {row_idx}, col: {col} — {repr(e)}")
        pm25 = pd.NA

    return pm25


def CONUS(sensors, name):
    sensors = sensors.copy()
    years = sensors["Time"].dt.year.unique()

    nc_files = []
    for year in years:
        nc_files.extend(glob.glob(f"./data/conus/*{year}*.nc"))

    ds = xr.open_mfdataset(
        nc_files,
        combine="nested",
        concat_dim="TSTEP",
        decode_cf=False
    )
    ds = decode_times(ds).load()        

    proj = Proj(
        proj="lcc",
        lat_1=ds.attrs["P_ALP"],
        lat_2=ds.attrs["P_BET"],
        lat_0=ds.attrs["YCENT"],
        lon_0=ds.attrs["XCENT"],
        x_0=0,
        y_0=0,
        ellps="sphere"
    )

    transformer = Transformer.from_proj("epsg:4326", proj, always_xy=True)

    lon_arr = sensors["Longitude"].to_numpy()
    lat_arr = sensors["Latitude"].to_numpy()
    x_arr, y_arr = transformer.transform(lon_arr, lat_arr)

    XORIG, YORIG = ds.attrs["XORIG"], ds.attrs["YORIG"]
    XCELL, YCELL = ds.attrs["XCELL"], ds.attrs["YCELL"]

    col = ((x_arr - XORIG) // XCELL).astype("int64")
    row = ((y_arr - YORIG) // YCELL).astype("int64")

    in_bounds = (
        (0 <= col) & (col < ds.dims["COL"]) &
        (0 <= row) & (row < ds.dims["ROW"])
    )

    time_idx = ds.indexes["time"].get_indexer(sensors["Time"].to_numpy(), method="nearest")
    time_ok = time_idx >= 0

    valid = in_bounds & time_ok

    pm25_grid = ds["PM25_TOT"].isel(LAY=0).values  

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

def fmt_ee_output(file):
    pass

def CAMS(sensors, name):
    sensors["CAMS"] = pd.NA

    data = pd.read_csv(f"cams_{name}", low_memory=False)

    data = data.rename(columns={
        "lat": "Latitude",
        "lon": "Longitude",
        "time": "Time",
    })

    data = data[["Latitude", "Longitude", "Time", "first"]]

    sensors = sensors.merge(data, on=["Latitude", "Longitude", "Time"], how="left")
    sensors["CAMS"] = sensors["first"]
    sensors = sensors.drop(columns=["first"])

    return sensors


def main() -> None:
    parse_args()
    try_auth()

    files = SENSOR_PATH.iterdir()
    sources = [CAMS]

    for file in files:
        print(f"Reading {file.name}")
        sensors = pd.read_csv(file, low_memory=False)
        for source in sources:
            sensors = fmt_sensors(sensors)
            sensors = source(sensors, file.name)
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

