import csv
import os
import rasterio
from rasterio.warp import transform

# Base path to your data
data_dir = "/home/home/Data"
fires_csv_path = os.path.join(data_dir, "fires.csv")
output_csv_path = os.path.join(data_dir, "fires_with_latlon.csv")

# Load all year directories
year_dirs = [os.path.join(data_dir, d) for d in os.listdir(data_dir) if d.isdigit()]

def find_fire_dir(fire_id):
    for year_dir in year_dirs:
        try:
            for d in os.listdir(year_dir):
                if fire_id in d:
                    return os.path.join(year_dir, d)
        except FileNotFoundError:
            continue
    return None

def get_lat_lon_from_tif(fire_dir):
    try:
        for fname in os.listdir(fire_dir):
            if fname.endswith(".tif"):
                tif_path = os.path.join(fire_dir, fname)
                with rasterio.open(tif_path) as src:
                    bounds = src.bounds
                    x_center = (bounds.left + bounds.right) / 2
                    y_center = (bounds.top + bounds.bottom) / 2

                    # Transform to lat/lon if needed
                    if src.crs.to_epsg() != 4326:
                        lon, lat = transform(
                            src.crs, 'EPSG:4326',
                            [x_center], [y_center]
                        )
                        return lat[0], lon[0]
                    else:
                        return y_center, x_center
    except Exception as e:
        print(f"Error reading tif in {fire_dir}: {e}")
    return None, None

# Process CSV
with open(fires_csv_path, newline="") as f_in, open(output_csv_path, "w", newline="") as f_out:
    reader = csv.DictReader(f_in)
    fieldnames = reader.fieldnames + ["lat", "lon"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        fire_id = row["fire_id"]
        fire_dir = find_fire_dir(fire_id)

        if fire_dir and os.path.exists(fire_dir):
            lat, lon = get_lat_lon_from_tif(fire_dir)
            if lat is None or lon is None:
                print(f"Warning: No valid .tif found for fire {fire_id}")
        else:
            print(f"Warning: No directory found for fire {fire_id}")
            lat, lon = None, None

        row["lat"] = lat
        row["lon"] = lon
        writer.writerow(row)
