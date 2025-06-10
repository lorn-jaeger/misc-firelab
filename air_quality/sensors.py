from pathlib import Path
import pandas as pd
import argparse
import warnings
from ee.geometry import Geometry
from ee.imagecollection import ImageCollection

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

def save(file):
    file.to_csv("data/out/out", index=False)

def try_auth():
    geemap.ee_initialize()

_cams_cache = {}

def get_CAMS_pm25(row):
    """
    Look up the CAMS PM2.5 value for a single sensor row.

    Assumes _cams_cache has been filled by CAMS() with a mapping:
        (lon, lat) -> DataFrame(time, pm25)
    """

    key = (row["longitude"], row["latitude"])
    df = _cams_cache.get(key)
    if df is None or df.empty:
        return pd.NA

    # Determine the timestamp for the sensor reading
    if "time" in row and pd.notna(row["time"]):
        t = row["time"]
    else:
        try:
            date = pd.to_datetime(row["Date GMT"])
            hour = (
                pd.to_datetime(row["Time GMT"], format="%H:%M").hour
                if isinstance(row["Time GMT"], str)
                else int(row["Time GMT"])
            )
            t = date + pd.to_timedelta(hour, unit="h")
        except Exception:
            return pd.NA

    match = df.loc[df["time"] == t, "pm25"]
    return match.iloc[0] if not match.empty else pd.NA


def CAMS(sensors):
    """
    Add a 'CAMS' column to the sensor DataFrame with ECMWF-CAMS PM2.5 values.
    """

    global _cams_cache
    _cams_cache = {}  # Reset for this batch

    # Standardize column names expected downstream
    if "latitude" not in sensors.columns and "Latitude" in sensors.columns:
        sensors = sensors.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})
    if "time" not in sensors.columns:
        sensors["Date GMT"] = pd.to_datetime(sensors["Date GMT"])
        sensors["Time GMT"] = (
            pd.to_datetime(sensors["Time GMT"], format="%H:%M").dt.hour
            if sensors["Time GMT"].dtype == object
            else sensors["Time GMT"].astype(int)
        )
        sensors["time"] = sensors["Date GMT"] + pd.to_timedelta(sensors["Time GMT"], unit="h")

    # Pull the CAMS imagery only for the span of this file
    band = "particulate_matter_d_less_than_25_um_surface"
    start = sensors["time"].min()
    end = sensors["time"].max()

    cams_ic = (
        ImageCollection("ECMWF/CAMS/NRT")
        .select(band)
        .filterDate(start, end)
    )

    # Download CAMS time–series for each unique sensor location
    coords = sensors[["longitude", "latitude"]].drop_duplicates()
    for _, r in coords.iterrows():
        point = Geometry.Point([r["longitude"], r["latitude"]])
        raw = cams_ic.getRegion(point, 10_000).getInfo()
        headers, data = raw[0], raw[1:]
        df = pd.DataFrame(data, columns=headers)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.rename(columns={band: "pm25"})
        df["pm25"] *= 1_000_000_000  # kg m⁻³ → µg m⁻³
        _cams_cache[(r["longitude"], r["latitude"])] = df[["time", "pm25"]]

    # Vectorised lookup of CAMS values for every sensor row
    sensors["CAMS"] = sensors.apply(get_CAMS_pm25, axis=1)
    return sensors

def CONUS(sensors):
    sensors["CONUS"] = pd.NA
    return sensors

def MERRA2(sensors):
    sensors["MEERA2"] = pd.NA
    return sensors

def MERRA2R(sensors):
    sensors["MEERA2R"] = pd.NA
    return sensors

def AIRNOW(sensors):
    sensors["AIRNOW"] = pd.NA
    return sensors

def main() -> None:
    parse_args()
    try_auth()

    files = SENSOR_PATH.iterdir()
    sources = [CAMS, CONUS, MERRA2, MERRA2R, AIRNOW]

    for file in files:
        print(f"Reading {file.name}")
        sensors = pd.read_csv(file, low_memory=False)
        for source in sources:
            sensors = source(sensors)
        save(sensors)

if __name__ == "__main__":
    main()



