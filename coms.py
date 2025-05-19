#!/usr/bin/env python3
"""
Time-align US EPA hour-resolution PM2.5 sensor data with ECMWF CAMS NRT PM2.5
forecasts for every wildfire location listed in the per-year YAML configs.

Output: data/fires/time_aligned_fire_analysis.csv
Columns:
    fire_lat, fire_lon,
    sensor_lat, sensor_lon,
    parameter_code,          # 88101 (FRM/FEM) or 88502 (non-FRM/FEM)
    measurement,             # observed μg m-3
    pm25_model,              # model kg m-3 (None if no image)
    year,                    # four-digit string
    init_time,               # yyyymmddHH (00 or 12 UTC run)
    forecast_hour            # 0-120
"""

import datetime as dt
from pathlib import Path

import ee
import geemap
import pandas as pd
import yaml
from tqdm import tqdm


# ----------------------------------------------------------------------
# 0.  constants
# ----------------------------------------------------------------------
BAND_PM25 = "particulate_matter_d_less_than_25_um_surface"
MAX_LEAD_HR = 120             # CAMS delivers 5-day (0-120 h) forecasts
SCALE_M = 10_000              # ≈ native 0.4° CAMS grid


# ----------------------------------------------------------------------
# 1.  Earth Engine helpers
# ----------------------------------------------------------------------
def ee_init() -> None:
    geemap.ee_initialize()


def build_cams_collection(init_dt: dt.datetime, lead_hr: int) -> ee.ImageCollection:

    return (ee.ImageCollection("ECMWF/CAMS/NRT")
            .filter(ee.Filter.eq("model_initialization_hour", init_dt.hour))
            .filter(ee.Filter.eq("model_forecast_hour", int(lead_hr))))


def get_cams_value(lat: float, lon: float,
                   init_dt: dt.datetime, lead_hr: int) -> float | None:

    col = build_cams_collection(init_dt, lead_hr)
    if col.size().getInfo() == 0:
        return None

    img = col.first().select(BAND_PM25)
    point = ee.Geometry.Point(lon, lat)

    try:
        val = (img.reduceRegion(ee.Reducer.first(), point, SCALE_M)
                   .getInfo()
                   .get(BAND_PM25))
    except Exception:
        val = None

    return val


# ----------------------------------------------------------------------
# 2.  Forecast-matching helpers
# ----------------------------------------------------------------------
def all_init_leads(obs_dt: dt.datetime) -> list[tuple[dt.datetime, int]]:
    pairs: list[tuple[dt.datetime, int]] = []
    step = dt.timedelta(hours=12)

    init_dt = obs_dt.replace(minute=0, second=0, microsecond=0,
                             hour=0 if obs_dt.hour < 12 else 12)

    while True:
        lead = int((obs_dt - init_dt).total_seconds() // 3600)
        if lead < 0 or lead > MAX_LEAD_HR:
            break
        pairs.append((init_dt, lead))
        init_dt -= step

    return pairs


def sample_all_forecasts(lat: float, lon: float,
                         obs_dt: dt.datetime) -> list[tuple[dt.datetime, int, float | None]]:
    out: list[tuple[dt.datetime, int, float | None]] = []
    for init_dt, lead in all_init_leads(obs_dt):
        val = get_cams_value(lat, lon, init_dt, lead)
        out.append((init_dt, lead, val))
    return out


# ----------------------------------------------------------------------
# 3.  Per-fire processing
# ----------------------------------------------------------------------
def process_fire(fire_cfg: dict, sensor_df: pd.DataFrame,
                 outfile: Path, year: str, buf: float) -> None:
    lat0, lon0 = fire_cfg["latitude"], fire_cfg["longitude"]

    sensors = sensor_df[
        (sensor_df["Latitude"].sub(lat0).abs() < buf) &
        (sensor_df["Longitude"].sub(lon0).abs() < buf)
    ]
    if sensors.empty:
        return

    for _, row in tqdm(sensors.iterrows(), total=len(sensors), leave=False):
        obs_dt = row["datetime_utc"].to_pydatetime()
        triplets = sample_all_forecasts(row.Latitude, row.Longitude, obs_dt)

        with outfile.open("a") as f:
            for init_dt, lead, pm25 in triplets:
                f.write(
                    f"{lat0},{lon0},"
                    f"{row.Latitude},{row.Longitude},"
                    f"{row.parameter_code},{row['Sample Measurement']},"
                    f"{pm25},{year},{init_dt:%Y%m%d%H},{lead}\n"
                )


# ----------------------------------------------------------------------
# 4.  Driver
# ----------------------------------------------------------------------
def main() -> None:
    ee_init()

    cfg_dir = Path("/home/home/Code/Fire/EE_Wildfire/config")
    sensor_dir = Path("data/sensor")
    out_dir = Path("data/fires")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "time_aligned_fire_analysis.csv"

    if not out_file.exists():
        out_file.write_text(
            "fire_lat,fire_lon,sensor_lat,sensor_lon,"
            "parameter_code,measurement,pm25_model,year,init_time,forecast_hour\n"
        )

    for cfg_path in cfg_dir.glob("us_fire_*_1e7.yml"):
        year = cfg_path.stem.split("_")[2]

        with cfg_path.open() as f:
            cfg = yaml.safe_load(f)

        sensor_frames = []
        for code in ("88101", "88502"):
            csv = sensor_dir / f"hourly_{code}_{year}.csv"
            if not csv.exists():
                continue
            df = pd.read_csv(
                csv,
                parse_dates={"datetime_utc": ["Date GMT", "Time GMT"]},
                dtype={"Latitude": float, "Longitude": float},
                low_memory=False,
            )
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            df["parameter_code"] = code
            sensor_frames.append(df)

        if not sensor_frames:
            print(f"No sensor data for {year}")
            continue

        sensors = pd.concat(sensor_frames, ignore_index=True)
        buf = cfg.get("rectangular_size", 0.5)

        fires = {k: v for k, v in cfg.items()
                 if k not in ("output_bucket", "rectangular_size", "year")}

        for name, fire_cfg in tqdm(fires.items(), desc=f"{year} fires"):
            try:
                process_fire(fire_cfg, sensors, out_file, year, buf)
            except Exception as e:
                print(f"{name}: {e}")


if __name__ == "__main__":
    main()
