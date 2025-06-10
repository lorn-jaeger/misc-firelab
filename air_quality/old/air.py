import csv

import geemap
from pathlib import Path
import yaml
import pandas as pd
import ee

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

def get_point_data(s: pd.DataFrame, fcams: ee.ImageCollection):
    unique_sensors = s[["longitude", "latitude"]].drop_duplicates()

    all_point_data = []

    for _, row in unique_sensors.iterrows():
        point = ee.Geometry.Point([row['longitude'], row['latitude']])
        raw = fcams.getRegion(point, 10_000).getInfo()
        headers = raw[0]
        data = raw[1:]
        df = pd.DataFrame(data, columns=headers)
        df["latitude"] = row["latitude"]
        df["longitude"] = row["longitude"]
        all_point_data.append(df)

    point_data = pd.concat(all_point_data, ignore_index=True)
    point_data["time"] = pd.to_datetime(point_data["time"], unit="ms")
    point_data.rename(columns={"particulate_matter_d_less_than_25_um_surface": "pm25"}, inplace=True)
    point_data["pm25"] *= 1_000_000_000
   # point_data = point_data[point_data["id"].str.contains("F000", na=False)]

    return point_data



def process_fire(fire_name: str, fire_data: dict, sensors: pd.DataFrame, cams: ee.ImageCollection, writer: csv.writer):
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

    region = ee.Geometry.Rectangle([
        longitude - 0.5,
        latitude - 0.5,
        longitude + 0.5,
        latitude + 0.5
    ])

    fcams = (
        cams
        .filterBounds(region)
        .filterDate(start, end)
    )

    point_data = get_point_data(s, fcams)

    merged = pd.merge(
        s,
        point_data,
        on=["latitude", "longitude", "time"],
        how="inner",
        suffixes=("_sensor", "_sat")
    )

    for _, row in merged.iterrows():
        writer.writerow([
            name,
            row["id_sensor"],
            row["id_sat"],
            row["latitude"],
            row["longitude"],
            row["time"],
            row["pm25_sensor"],
            row["pm25_sat"],
        ])
        count += 1

    print(f"Done processing {name}, {count} matches")


def main() -> None:
    geemap.ee_initialize()

    fire_dir = Path("data/fires")
    sensor_dir = Path("data/sensor")
    out_path = Path("fout.csv")

    cams = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .select("particulate_matter_d_less_than_25_um_surface")
    )

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
            ]
        )

        for fire_path in fire_dir.glob("us_fire_*_1e7.yml"):
            year = fire_path.stem.split("_")[2]
            for t in ["88101", "88502"]:
                file = f"{sensor_dir}/hourly_{t}_{year}.csv"
                print(f"Reading {file}")
                sensors = pd.read_csv(file, header=0, skiprows=0)
                sensors = fmt_sensor_data(sensors)
                print(f"Processing {file}")
                with fire_path.open() as f:
                    fire_file = yaml.safe_load(f)
                    fires = {k: v for k, v in fire_file.items() if k not in ("output_bucket", "rectangular_size", "year")}
                    for fire_name, fire_data in fires.items():
                        try:
                            process_fire(fire_name, fire_data, sensors, cams, writer)
                        except Exception as e:
                            print(e)


if __name__ == "__main__":
    main()



