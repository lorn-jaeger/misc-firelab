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
    key = (row["longitude"], row["latitude"])
    df = _cams_cache.get(key)
    if df is None or df.empty:
        return pd.NA

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
    global _cams_cache
    _cams_cache = {}

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

    band = "particulate_matter_d_less_than_25_um_surface"
    start = sensors["time"].min()
    end = sensors["time"].max()

    cams_ic = (
        ImageCollection("ECMWF/CAMS/NRT")
        .select(band)
        .filterDate(start, end)
    )

    coords = sensors[["longitude", "latitude"]].drop_duplicates()
    for _, r in coords.iterrows():
        point = Geometry.Point([r["longitude"], r["latitude"]])
        raw = cams_ic.getRegion(point, 10_000).getInfo()
        headers, data = raw[0], raw[1:]
        df = pd.DataFrame(data, columns=headers)
        df["time"] = pd.to_datetime(df["time"], unit="ms")
        df = df.rename(columns={band: "pm25"})
        df["pm25"] *= 1_000_000_000  
        _cams_cache[(r["longitude"], r["latitude"])] = df[["time", "pm25"]]

    sensors["CAMS"] = sensors.apply(get_CAMS_pm25, axis=1)
    return sensors

def CONUS(sensors):
    sensors["CONUS"] = pd.NA
    return sensors


# ---------------------------------------------------------------------
# MERRA-2 helper – download once per file and cache per-location look-ups
# ---------------------------------------------------------------------
import pandas as pd
import xarray as xr
import earthaccess

# in-memory caches reused by every call during one CSV pass
_merra_cache   = {}          # (lon, lat) ➜ DataFrame(time, pm25)
_merra_dataset = None        # the xarray.Dataset covering the current time span

def _load_merra_dataset(start, end):
    """
    Download all MERRA-2 CNN PM₂.₅ granules spanning [start, end]
    and return an xarray.Dataset.  Reuses the previous dataset
    if it already covers the requested window.
    """
    global _merra_dataset, _merra_start, _merra_end

    # If we already have a dataset that fully brackets the request, reuse it

    # Otherwise fetch the new range
    _merra_start, _merra_end = start, end

    earthaccess.login(persist=True)

    results = earthaccess.search_data(
        concept_id="C3094710982-GES_DISC",     # MERRA-2 CNN Surface PM₂.₅
        temporal=(start, end),
    )
    files = earthaccess.download(results)
    if not files:
        _merra_dataset = None
        return None

    # open_mfdataset with by_coords makes time the concat dim automatically
    _merra_dataset = xr.open_mfdataset(files, combine="by_coords")
    return _merra_dataset


# ---------------------------------------------------------------------
# Per-row accessor (used by sensors.apply below)
# ---------------------------------------------------------------------
def _get_MERRA2_pm25(row):
    key = (row["longitude"], row["latitude"])
    df  = _merra_cache.get(key)
    if df is None or df.empty:
        return pd.NA
    match = df.loc[df["time"] == row["time"], "pm25"]
    return match.iloc[0] if not match.empty else pd.NA


# ---------------------------------------------------------------------
# Public function plugged into the pipeline:  MERRA2(sensors)
# ---------------------------------------------------------------------
def MERRA2(sensors):
    """
    Add a **MERRA2** column containing CNN Surface PM₂.₅ (µg m⁻³)
    collocated in space & time with every sensor reading.

    Works exactly like the CAMS helper you already integrated.
    """
    global _merra_cache
    _merra_cache = {}            # reset for this CSV batch

    # Ensure canonical columns exist (identical to the CAMS helper)
    if "time" not in sensors.columns:
        sensors["Date GMT"] = pd.to_datetime(sensors["Date GMT"])
        sensors["Time GMT"] = (
            pd.to_datetime(sensors["Time GMT"], format="%H:%M")
              .dt.hour if sensors["Time GMT"].dtype == object
              else sensors["Time GMT"].astype(int)
        )
        sensors["time"] = sensors["Date GMT"] + pd.to_timedelta(
            sensors["Time GMT"], unit="h"
        )
    if "latitude" not in sensors.columns:
        sensors = sensors.rename(columns={"Latitude": "latitude",
                                          "Longitude": "longitude"})

    if sensors.empty:
        sensors["MERRA2"] = pd.NA
        return sensors

    # 1️⃣  Load (or reuse) a dataset covering the full sensor time span
    start, end = sensors["time"].min(), sensors["time"].max()
    ds = _load_merra_dataset(start, end)
    if ds is None:
        sensors["MERRA2"] = pd.NA
        return sensors

    # 2️⃣  Pick the first matching PM₂.₅ variable name
    var_name_candidates = [
        "MERRA2_CNN_Surface_PM25",   # current
        "MERRA2_CNN_PM25",           # legacy
        "pm25",                      # fallback
    ]
    for vn in var_name_candidates:
        if vn in ds.data_vars:
            pm25_da = ds[vn]
            break
    else:   # no variable found
        sensors["MERRA2"] = pd.NA
        return sensors

    # 3️⃣  Build per-location time-series caches
    unique_sites = sensors[["longitude", "latitude"]].drop_duplicates()
    for _, site in unique_sites.iterrows():
        lon, lat = site["longitude"], site["latitude"]

        # nearest-cell slice (lat/lon dims are 'lat','lon' in these granules)
        ts = (
            pm25_da.sel(lat=lat, lon=lon, method="nearest")
                   .to_dataframe()
                   .reset_index()[["time", vn]]
        )
        ts = ts.rename(columns={vn: "pm25"}) #type: ignore
        _merra_cache[(lon, lat)] = ts

    # 4️⃣  Vectorised lookup for every row
    sensors["MERRA2"] = sensors.apply(_get_MERRA2_pm25, axis=1)
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



