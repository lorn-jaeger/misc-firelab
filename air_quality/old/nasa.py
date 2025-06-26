import xarray as xr
import earthaccess
import csv
from pathlib import Path
import yaml
import pandas as pd
import ee
import shutil

# pyright: ignore-all


def fmt_sensor_data(sensors: pd.DataFrame):
    sensors["Date GMT"] = pd.to_datetime(sensors["Date GMT"])
    sensors["Time GMT"] = pd.to_datetime(sensors["Time GMT"], format="%H:%M").dt.hour
    sensors["time"] = sensors["Date GMT"] + pd.to_timedelta(sensors["Time GMT"], unit="h")
    sensors.rename(columns={"Latitude": "latitude"}, inplace=True)
    sensors.rename(columns={"Longitude": "longitude"}, inplace=True)
    sensors.rename(columns={"Sample Measurement": "pm25"}, inplace=True)
    sensors["id"] = (
        sensors["State Code"].astype(str) +
        sensors["County Code"].astype(str) +
        sensors["Site Num"].astype(str)
    )

    sensors = sensors[["id", "latitude", "longitude", "time", "pm25"]]

    return sensors

def get_point_data(s: pd.DataFrame):
    return





def process_fire(fire_name: str, fire_data: dict, sensors: pd.DataFrame, writer: csv.writer, code):
    merra_data_dir = Path("/home/home/Code/FireLab/EE_Wildfire/air_quality/data")
    for path in merra_data_dir.iterdir():
        if path.is_dir() and "2025" in path.name:
            shutil.rmtree(path)

    name = fire_name
    latitude = fire_data.get("latitude")
    longitude = fire_data.get("longitude")
    start = pd.to_datetime(fire_data.get("start"))
    end = pd.to_datetime(fire_data.get("end"))

    print(f"Processing {name}")
    count = 0

    s = sensors[
        (abs(sensors["latitude"] - latitude) < 0.5) &
        (abs(sensors["longitude"] - longitude) < 0.5) &
        (sensors["time"] >= start) &
        (sensors["time"] <= end)
    ]

    if s.empty:
        print("No sensor data available")
        return

    results = earthaccess.search_data(
        concept_id="C3094710982-GES_DISC",  # Verify correct CAMS concept ID
        temporal=(start, end),
        bounding_box=(longitude - 0.5, latitude - 0.5, longitude + 0.5, latitude + 0.5),
    )
    files = earthaccess.download(results)

    # Open dataset and process
    ds = xr.open_mfdataset(files, combine="by_coords")
    pm25_var = "MERRA2_CNN_Surface_PM25"  # Confirm variable name in dataset

    # Convert sensor times to numpy datetime64
    s["time_np"] = s["time"].astype("datetime64[ns]")

    # Extract satellite data at sensor points
    try:
        sat_data = ds[pm25_var].interp(
            time=xr.DataArray(s["time_np"], dims="sensor"),
            lat=xr.DataArray(s["latitude"], dims="sensor"),
            lon=xr.DataArray(s["longitude"], dims="sensor")
        )
        s["pm25_sat"] = sat_data.values  # Adjust scaling if needed

    except KeyError as e:
        print(f"Variable {pm25_var} not found: {e}")
        return

    # Write merged results
    for _, row in s.iterrows():
        writer.writerow([
            name,
            row["id"],
            "CAMS",  # Placeholder for satellite ID
            row["latitude"],
            row["longitude"],
            row["time"], 
            row["pm25"],
            row["pm25_sat"],
            code,
        ])
        count += 1
    print(f"Done processing {name}, {count} matches")


def main() -> None:
    auth = earthaccess.login(persist=True)

    fire_dir = Path("data/fires")
    sensor_dir = Path("data/sensor")
    out_path = Path("data/out/out2.csv")


    with out_path.open("w", newline="", buffering=100) as f_out:
        writer = csv.writer(f_out)
        writer.writerow(
            [
                "fire_name",
                "sensor_id",
                "sat_id",
                "lat",
                "lon",
                "time",
                "sensor_pm25",
                "sat_pm25",
                "code"
            ]
        )

        for fire_path in fire_dir.glob("us_fire_*_1e7.yml"):
            year = fire_path.stem.split("_")[2]
            for t in ["88101", "88502"]:
                file = f"{sensor_dir}/hourly_{t}_{year}.csv"
                print(f"Reading {file}")
                sensors = pd.read_csv(file, header=0, skiprows=0, low_memory=False)
                sensors = fmt_sensor_data(sensors)
                print(f"Processing {file}")
                with fire_path.open() as f:
                    fire_file = yaml.safe_load(f)
                    fires = {k: v for k, v in fire_file.items() if k not in ("output_bucket", "rectangular_size", "year")}
                    for fire_name, fire_data in fires.items():
                        try:
                            process_fire(fire_name, fire_data, sensors, writer, t)
                        except Exception as e:
                            print(e)


if __name__ == "__main__":
    main()
