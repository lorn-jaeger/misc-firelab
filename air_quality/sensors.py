from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import argparse
import warnings
from pyproj import Proj, Transformer
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

SENSOR_PATH = Path("data/sensors/using")
TEST_PATH = Path("data/test/sensors")
OUT_PATH = Path("data/out")


def clear_output():
    for file in OUT_PATH.iterdir():
        if file.is_file():
            file.unlink()


def parse_args():
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

def fmt_sensors(sensors):
     sensors['Date GMT'] = pd.to_datetime(sensors['Date GMT'])
     sensors['Time'] = sensors['Date GMT'] + pd.to_timedelta(sensors['Time GMT'] + ':00')

     sensors["CONUS"] = pd.NA
     sensors["MERRA2"] = pd.NA
     sensors["MERRA2R"] = pd.NA
     sensors["CAMS"] = pd.NA
 
     sensors = sensors[["Time", "Latitude", "Longitude", "Sample Measurement", "CONUS", "MERRA2", "MERRA2R", "CAMS"]]

     return sensors


def get_unique(sensors):
    result = sensors.groupby(['Latitude', 'Longitude'])['Time'].agg(['min', 'max']).reset_index()
    result.rename(columns={'min': 'Start Time', 'max': 'End Time'}, inplace=True)

    return result


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

import numpy as np

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
    pm25_interp = ds[var].interp(
        time=xr.DataArray(sensors["time_np"], dims="sensor"),
        lat=xr.DataArray(sensors["lat"], dims="sensor"),
        lon=xr.DataArray(sensors["lon"], dims="sensor"),
        method="nearest"
    ).values

    qflag_interp = ds["QFLAG"].interp(
        lat=xr.DataArray(sensors["lat"], dims="sensor"),
        lon=xr.DataArray(sensors["lon"], dims="sensor"),
        method="nearest"
    ).values

    high_quality = (qflag_interp == 3) | (qflag_interp == 4)
    pm25_interp[~high_quality] = np.nan  

    sensors["MERRA2R"] = pm25_interp

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
    year = sdate // 1000
    doy = sdate % 1000
    start_time = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)

    print(sdate, year, doy, start_time, name)

    n_times = ds.sizes["TSTEP"]
    datetimes = [start_time + timedelta(hours=i) for i in range(n_times)]


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

    x, y = transformer.transform(sensors["Longitude"].to_numpy(), sensors["Latitude"].to_numpy())

    col = ((x - ds.XORIG) / ds.XCELL).astype(int)
    row = ((y - ds.YORIG) / ds.YCELL).astype(int)

    in_bounds = (
        (0 <= col) & (col < ds.dims["COL"]) &
        (0 <= row) & (row < ds.dims["ROW"])
    )

    ds = ds.swap_dims({"TSTEP": "time"})

    time_idx = ds.indexes["time"].get_indexer(sensors["Time"].to_numpy())

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


def main() -> None:
    parse_args()

    files = SENSOR_PATH.iterdir()
    sources = [MERRA2R]

    for file in files:
        print(f"Reading {file.name}")
        sensors = pd.read_csv(file, low_memory=False)
        sensors = fmt_sensors(sensors)
        print(f"Processing {file.name}")
        for source in sources:
            print(f"{source.__name__}...")
            try:
                sensors = source(sensors, file.name)
            except Exception as e:
                print(f"Error processing {source.__name__} for {file.name}: {e}")
                continue
            print(sensors)      # 
        save(sensors, file.name)

if __name__ == "__main__":
    main()


