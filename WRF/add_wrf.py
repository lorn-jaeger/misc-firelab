
import rasterio
from rasterio.transform import xy
from rasterio.warp import transform, transform_bounds
from pathlib import Path
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from pyproj import CRS
import numpy as np

tiffs = Path("./data/using/tiffs/")
wrf = Path("./data/using/wrf/")
out = Path("./data/using/out/")

def main():
    tif_path = tiffs / "Image_Export_fire_20777134_2017-07-15.tif"
    wrf_path = wrf / "wrfout_d01_2017-07-15_00:00:00"

    # --- Open GeoTIFF ---
    with rasterio.open(tif_path) as src:
        print("Number of bands:", src.count)
        print("Width, Height:", src.width, src.height)
        print("CRS (projection):", src.crs)
        print("Transform (affine):", src.transform)
        print("Resolution:", src.res) 
        print("Data type:", src.dtypes)
        print("Bounds:", src.bounds)
        print("Driver:", src.driver)

        width = src.width
        height = src.height
        dst_crs = src.crs

        center_row = height // 2
        center_col = width // 2
        utm_lon, utm_lat = xy(src.transform, center_row, center_col)
        lon, lat = transform(src.crs, "EPSG:4326", [utm_lon], [utm_lat])  # type: ignore
        print("Center Lat/Lon:", lat[0], lon[0])

        original_bands = src.read()
        profile = src.profile

        bounds_latlon = transform_bounds(dst_crs, "EPSG:4326", *src.bounds)

    # --- Load WRF T2 variable ---
    ds = xr.open_dataset(wrf_path)
    da = ds["T2"].isel(Time=0)

    # Rename spatial dims
    da = da.rename({"west_east": "x", "south_north": "y"})
    da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)

    # Create x/y coordinate arrays from DX/DY and shape
    dx = float(ds.attrs["DX"])
    dy = float(ds.attrs["DY"])
    nx = da.sizes["x"]
    ny = da.sizes["y"]
    x_coords = (np.arange(nx) - nx // 2) * dx
    y_coords = (np.arange(ny) - ny // 2) * dy
    da = da.assign_coords(x=("x", x_coords), y=("y", y_coords))

    # Construct Lambert CRS
    lcc_crs = CRS.from_dict({
        "proj": "lcc",
        "lat_1": float(ds.attrs.get("TRUELAT1", ds.get("TRUELAT1"))),
        "lat_2": float(ds.attrs.get("TRUELAT2", ds.get("TRUELAT2"))),
        "lat_0": float(ds.attrs.get("MOAD_CEN_LAT", ds.get("MOAD_CEN_LAT"))),
        "lon_0": float(ds.attrs.get("CEN_LON", ds.get("CEN_LON"))),
        "x_0": 0,
        "y_0": 0,
        "datum": "WGS84",
        "units": "m"
    })
    da.rio.write_crs(lcc_crs.to_wkt(), inplace=True)

    # --- Clip and reproject to match TIFF ---
    da_clip = da.rio.clip_box(*bounds_latlon, allow_one_dimensional_raster=True)
    tif_rio = rxr.open_rasterio(tif_path, masked=True)
    da_match = da_clip.rio.reproject_match(tif_rio, resampling=Resampling.bilinear)

    # --- Add new band ---
    new_band = da_match.data.astype(profile["dtype"], copy=False)
    profile.update(count=profile["count"] + 1)

    out_path = out / "tiff_with_wrf.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        for i in range(profile["count"] - 1):
            dst.write(original_bands[i], i + 1)
        dst.write(new_band[0], profile["count"])
        dst.set_band_description(profile["count"], "T2_from_WRF")

    print("Output written to:", out_path)

if __name__ == '__main__':
    main()

