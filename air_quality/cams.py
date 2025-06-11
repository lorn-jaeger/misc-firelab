import pandas as pd
from ee.imagecollection import ImageCollection
from ee.geometry import Geometry
from sensors import get_unique

def parse_gee_region(raw):
    headers = raw[0]
    data = raw[1:]

    df = pd.DataFrame(data, columns=headers)

    df = df[df["time"].apply(lambda x: isinstance(x, (int, float)))]

    df["Time"] = pd.to_datetime(df["time"], unit="ms")
    df["Longitude"] = df["longitude"].astype(float)
    df["Latitude"] = df["latitude"].astype(float)
    df["pm25"] = df["particulate_matter_d_less_than_25_um_surface"].astype(float)

    df["pm25"] *= 1_000_000_000

    return df[["Time", "Longitude", "Latitude", "pm25"]]

def CAMS(sensors):
    sensors["CAMS"] = pd.NA

    cams = (
        ImageCollection("ECMWF/CAMS/NRT")
        .select("particulate_matter_d_less_than_25_um_surface")
    )

    cams_data = pd.DataFrame(columns=["Time", "Longitude", "Latitude", "pm25"]) #type: ignore

    locations = get_unique(sensors)

    # can use apply instead but this is only 1000 iterations
    # most of the time is taken by api calls to earth engine
    for _, row in locations.iterrows():
        point = Geometry.Point(row["Longitude"], row["Latitude"])
        # batching is needed so we don't hit the 3000 item limit
        months = pd.date_range(pd.to_datetime(row["Start Time"]), pd.to_datetime(row["End Time"]), freq="ME")

        for month in months:
            start = month
            end = month + pd.DateOffset(months=1)

            # skip if start is before the start of the dataset
            if start < pd.Timestamp("2016-06-23"):
                continue

            monthly_data = (
                cams
                .filterBounds(point)
                .filterDate(start, end)
                .getRegion(point, scale=1)
                .getInfo()
            )
            monthly_data = parse_gee_region(monthly_data)
            # Needed overwrite. Google Earth Engine rounds lats and lons so there are slightly off
            # Otherwise there will be no matches on merge
            monthly_data["Latitude"] = row["Latitude"]
            monthly_data["Longitude"] = row["Longitude"]
            cams_data = pd.concat([cams_data, monthly_data], ignore_index=True) #type: ignore

    sensors = sensors.merge(
        cams_data,
        on=["Time", "Latitude", "Longitude"],
        how="left"
    )

    sensors["CAMS"] = sensors["pm25"]
    sensors.drop(columns=["pm25"], inplace=True)

    return sensors


