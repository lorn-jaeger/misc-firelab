
#!/usr/bin/env python3
"""
Streaming satellite‑join pipeline for large EPA hourly PM₂.₅ archives
-------------------------------------------------------------------
• **Input**   : one or many `hourly_*.csv` files (≈ 40 GB total) in `data/sensor/`
• **Output**  : single CSV `data/out/sensors_with_sat.csv` with columns
               `id, lat, lon, time, pm25, pm25_sat`
• **Satellite**: MERRA‑2 CNN surface PM₂.₅ (C3094710982‑GES_DISC) — no CAMS / Earth Engine
• **Method**  :
    – stream‑read each sensor CSV in 1 M‑row chunks (≈100 MB) → constant RAM
    – derive cube key (1° × 1° × month) vectorised in NumPy
    – group rows by cube, download / cache MERRA‑2 granules once per cube,
      interpolate with `xarray` → `pm25_sat`
    – append annotated rows to the output file immediately (no global concat)
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv

import numpy as np
import pandas as pd
import xarray as xr
import tqdm
import earthaccess

###############################################################################
# 1  Configuration
###############################################################################
# Location of raw EPA CSVs and output
SENSOR_DIR   = Path("data/sensors")
OUT_CSV      = Path("data/out/sensors_with_sat.csv")
CACHE_DIR    = Path("data/merra_cache")

CHUNK_ROWS   = 1_000_000        # ≈100 MB/chunk; adjust to fit RAM
DLAT = DLON  = 1.0              # cube size in degrees

# MERRA‑2 collection + variable
MERRA_CONCEPT = "C3094710982-GES_DISC"          # NASA GES DISC concept ID
PM25_VAR      = "MERRA2_CNN_Surface_PM25"       # NetCDF variable name

earthaccess.login(persist=True)

###############################################################################
# 2  Helper functions
###############################################################################

def fmt_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise one raw chunk to columns id, lat, lon, time, pm25."""
    cols = {c.lower().strip(): c for c in df.columns}
    df.rename(columns=cols, inplace=True)
    df.rename(
        columns={
            "Date GMT": "date_gmt",
            "Time GMT": "time_gmt",
            "Sample Measurement": "pm25",
            "Latitude": "lat",
            "Longitude": "lon",
        },
        inplace=True,
    )
    df["date_gmt"] = pd.to_datetime(df["date_gmt"], errors="coerce")
    df = df[df["date_gmt"].notna()]
    df["time"] = df["date_gmt"] + pd.to_timedelta(df["time_gmt"] + ":00")
    df["id"] = (
        df["State Code"].astype(str).str.zfill(2)
        + df["County Code"].astype(str).str.zfill(3)
        + df["Site Num"].astype(str).str.zfill(4)
    )
    return df[["id", "lat", "lon", "time", "pm25"]]

def cube_key_vec(lat: np.ndarray, lon: np.ndarray, ts: pd.Series) -> np.ndarray:
    """Vectorised cube key → structured ndarray (row,col,YYYYMM)."""
    row = np.floor((lat + 90) / DLAT).astype(int)
    col = np.floor((lon + 180) / DLON).astype(int)
    ym  = ts.dt.strftime("%Y%m")
    return np.char.add(
        np.char.add(row.astype(str), "_" + col.astype(str) + "_"), ym.values
    )

def fetch_merra(lat_min, lat_max, lon_min, lon_max, t0, t1):
    bbox = (lon_min - 0.1, lat_min - 0.1, lon_max + 0.1, lat_max + 0.1)
    results = earthaccess.search_data(
        concept_id=MERRA_CONCEPT,
        temporal=(t0, t1),
        bounding_box=bbox,
    )
    files = earthaccess.download(results)
    return xr.open_mfdataset(files, combine="by_coords")[PM25_VAR]

def annotate_cube(df_cube: pd.DataFrame) -> pd.Series:
    """Return interpolated pm25_sat for rows in df_cube."""
    lat_min, lat_max = df_cube["lat"].min(), df_cube["lat"].max()
    lon_min, lon_max = df_cube["lon"].min(), df_cube["lon"].max()
    t0 = df_cube["time"].min().floor("D")
    t1 = (df_cube["time"].max() + pd.Timedelta(days=1)).floor("D")

    sat = fetch_merra(lat_min, lat_max, lon_min, lon_max, t0, t1)

    pm25_sat = sat.interp(
        lon=xr.DataArray(df_cube["lon"].values, dims="points"),
        lat=xr.DataArray(df_cube["lat"].values, dims="points"),
        time=xr.DataArray(df_cube["time"].astype("datetime64[ns]").values, dims="points"),
    )
    return pm25_sat.values.astype(np.float32)

###############################################################################
# 3  Stream‑processing driver
###############################################################################

def process_file(csv_path: Path, csv_writer):
    """Stream‑read `csv_path`, annotate chunks, append to writer."""
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_ROWS, low_memory=False):
        chunk = fmt_chunk(chunk)
        # Cube key vectorised
        chunk["cube"] = cube_key_vec(chunk["lat"].values, chunk["lon"].values, chunk["time"])
        # Group by cube inside the chunk
        for cube_id, df_cube in chunk.groupby("cube"):
            try:
                df_cube = df_cube.copy()  # prevent SettingWithCopy
                df_cube["pm25_sat"] = annotate_cube(df_cube)
                csv_writer.writerows(
                    df_cube[["id", "lat", "lon", "time", "pm25", "pm25_sat"]].itertuples(index=False, name=None)
                )
            except Exception as e:
                print(f"⚠ cube {cube_id} failed: {e}")


def main():
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", buffering=1) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["id", "lat", "lon", "time", "pm25", "pm25_sat"])

        files = sorted(SENSOR_DIR.glob("hourly_*.csv"))
        for csv_path in tqdm.tqdm(files, desc="Sensor files"):
            process_file(csv_path, writer)


if __name__ == "__main__":
    main()
