import geemap
from pathlib import Path
import yaml
import pandas as pd
import ee

def process_fire(fire_name: str, fire_data: dict, sensors: pd.DataFrame, cams: ee.ImageCollection):
    name = fire_name
    latitude = fire_data.get("latitude")
    longitude = fire_data.get("longitude")
    start = pd.to_datetime(fire_data.get("start"))
    end = pd.to_datetime(fire_data.get("end"))
    scale = 10_000

    sensors["Date GMT"] = pd.to_datetime(sensors["Date GMT"])
    sensors["Time GMT"] = pd.to_datetime(sensors["Time GMT"], format="%H:%M").dt.hour

    sensors = sensors[
        (abs(sensors["Latitude"] - latitude) < 0.5) &
        (abs(sensors["Longitude"] - longitude) < 0.5) &
        (sensors["Date GMT"] >= start) &
        (sensors["Date GMT"] <= end)
    ]

    if sensors.empty:
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
    for _, row in sensors.iterrows():
        lat, long = row["Latitude"], row["Longitude"]
        date, hour = row["Date GMT"], row["Time GMT"]
        measurement = row["Sample Measurement"]

        # THIS IS REVERSED (x, y not lat, long)
        point = ee.Geometry.Point(long, lat)

        raw = fcams.getRegion(point, scale).getInfo()
        point_data = pd.DataFrame(raw)

        p_lat, p_long = point_data["latitude"], point_data["longitude"]
        p_pm25 = point_data["particulate_matter_d_less_than_25_um_surface"]
        p_info = point_data["id"]
        p_time = point_data["time"]





def main() -> None:
    geemap.ee_initialize()

    fire_dir = Path("data/fires")
    sensor_dir = Path("data/sensor")
    out_dir = Path("data/out")


    cams = (
        ee.ImageCollection("ECMWF/CAMS/NRT")
        .select("particulate_matter_d_less_than_25_um_surface")
    )

    for fire_path in fire_dir.glob("us_fire_*_1e7.yml"):
        year = fire_path.stem.split("_")[2]
        sensors = pd.read_csv(f"{sensor_dir}/hourly_88101_{year}.csv", header=0, skiprows=0)
        with fire_path.open() as f:
            fire_file = yaml.safe_load(f)
            fires = {k: v for k, v in fire_file.items() if k not in ("output_bucket", "rectangular_size", "year")}
            for fire_name, fire_data in fires.items():
                process_fire(fire_name, fire_data, sensors, cams)


if __name__ == "__main__":
    main()



