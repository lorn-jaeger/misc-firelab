import csv

import geemap
from pathlib import Path
import yaml
import pandas as pd
import ee

def process_fire(fire_name: str, fire_data: dict, sensors: pd.DataFrame, cams: ee.ImageCollection, writer: csv.writer):
    name = fire_name
    latitude = fire_data.get("latitude")
    longitude = fire_data.get("longitude")
    start = pd.to_datetime(fire_data.get("start"))
    end = pd.to_datetime(fire_data.get("end"))
    scale = 10_000

    print(f"Processing {name}")
    count = 0

    s = sensors[
        (abs(sensors["Latitude"] - latitude) < 0.5) &
        (abs(sensors["Longitude"] - longitude) < 0.5) &
        (sensors["Date GMT"] >= start) &
        (sensors["Date GMT"] <= end)
    ]

    if s.empty:
        return

    region = ee.Geometry.Rectangle([
        longitude - 0.5,
        latitude - 0.5,
        longitude + 0.5,
        latitude + 0.5
    ])

    fcams = (
        cams
        .filterBounds(region)
        .filterDate(start, end) # might be exclusive? will check later
    )

    rows = []
    for _, row in s.iterrows():
        lat, long = row["Latitude"], row["Longitude"]
        date, hour = row["Date GMT"], row["Time GMT"]
        measurement = row["Sample Measurement"]

        # THIS IS REVERSED (x, y not lat, long)
        point = ee.Geometry.Point(long, lat)

        raw = fcams.getRegion(point, scale).getInfo()
        headers = raw[0]
        data = raw[1:]
        point_data = pd.DataFrame(data, columns=headers)
        point_data["time"] = pd.to_datetime(point_data["time"], unit="ms")
        point_data.rename(columns={"particulate_matter_d_less_than_25_um_surface": "pm25"}, inplace=True)
        point_data["pm25"] *= 1_000_000_000

        w_data = point_data[point_data["time"] == row["time"]]

        for _, sat in w_data.iterrows():
            is_observation = "F000" in sat["id"]
            if is_observation:
                writer.writerow(
                    [
                        name,
                        str(row["State Code"]) + str(row["County Code"]) + str(row["Site Num"]),
                        sat["id"],
                        lat,
                        long,
                        sat["time"],
                        measurement,
                        sat["pm25"],
                    ]
                )
                count += 1

    print(f"Done processing {name}, {count} matches")


def main() -> None:
    geemap.ee_initialize()

    fire_dir = Path("data/fires")
    sensor_dir = Path("data/sensor")
    out_path = Path("out.csv")


    cams = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .select("particulate_matter_d_less_than_25_um_surface")
    )

    with out_path.open("w", newline="", buffering=1) as f_out:
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
            ]
        )

        for fire_path in fire_dir.glob("us_fire_*_1e7.yml"):
            year = fire_path.stem.split("_")[2]
            for t in ["88101", "88502"]:
                file = f"{sensor_dir}/hourly_{t}_{year}.csv"
                print(f"Reading {file}")
                sensors = pd.read_csv(file, header=0, skiprows=0)
                sensors["Date GMT"] = pd.to_datetime(sensors["Date GMT"])
                sensors["Time GMT"] = pd.to_datetime(sensors["Time GMT"], format="%H:%M").dt.hour
                sensors["time"] = sensors["Date GMT"] + pd.to_timedelta(sensors["Time GMT"], unit="h")
                print(f"Processing {file}")
                with fire_path.open() as f:
                    fire_file = yaml.safe_load(f)
                    fires = {k: v for k, v in fire_file.items() if k not in ("output_bucket", "rectangular_size", "year")}
                    for fire_name, fire_data in fires.items():
                        process_fire(fire_name, fire_data, sensors, cams, writer)


if __name__ == "__main__":
    main()



