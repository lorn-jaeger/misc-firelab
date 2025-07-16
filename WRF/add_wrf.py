import rasterio
import traceback
from wrf import getvar, enable_xarray   
import re
import xarray as xr
import rioxarray
import numpy as np
from pathlib import Path
import argparse
from dataclasses import dataclass
import datetime

enable_xarray()

WRF_PATH = Path("data/using/wrf")
TIFF_PATH = Path("data/using/tiffs")
OUT_PATH = Path("data/using/out")

@dataclass
class Fire:
    id: int
    tiff_path: Path
    wrf_path: Path
    
    @classmethod
    def from_filename(cls, dir_name):
        tiff_path = TIFF_PATH / dir_name
        wrf_path = WRF_PATH / dir_name
        stem = Path(dir_name).stem
        print(stem)
        parts = stem.split('_')
        print(parts)
        id = int(parts[1])

        return cls(id=id, tiff_path=tiff_path, wrf_path=wrf_path)


def clear_output():
    for file in OUT_PATH.iterdir():
        if file.is_file():
            file.unlink()

def parse_args():
    global WRF_PATH
    global TIFF_PATH
    global OUT_PATH

    parser = argparse.ArgumentParser()

    parser.add_argument('--wrf-path', type=Path, required=False)
    parser.add_argument('--tiff-path', type=Path, required=False)
    parser.add_argument('--out-path', type=Path, required=False)
    parser.add_argument('--clear-out', action='store_true', required=False)

    args = parser.parse_args()

    if args.wrf_path:
        WRF_PATH = args.wrf_path
    if args.tiff_path:
        TIFF_PATH = args.tiff_path
    if args.out_path:
        OUT_PATH = args.out_path

    if args.clear_out:
        clear_output() 

def get_tiff_date(name):
    return re.findall(r'\d{4}-\d{2}-\d{2}', name)

def get_tiff_ds(fire, tiff):
    with rasterio.open(tiff) as src:
        profile = src.profile.copy()
        data = src.read()

    date = get_tiff_date(tiff.name)[0]

    wrf_files = sorted(
        fire.wrf_path.glob(f"wrfout_d01_{date}_*:00:00")
    )

    ds = xr.open_mfdataset(
        wrf_files, #type: ignore
        combine="by_coords", 
        engine="netcdf4"
    )

    ds["Humidity"] = getvar(ds, "rh")
    ds["Precipitation"] = ds["RAINNC"] + ds["RAINC"]
    


    lats = ds["XLAT"].isel(Time=0).rename({"south_north": "y", "west_east": "x"})
    lons = ds["XLONG"].isel(Time=0).rename({"south_north": "y", "west_east": "x"})
    x = lons.isel(y=0).data
    y = lats.isel(x=0).data

    return ds, x, y, data, profile
    

def append_wrf(fire, tiff):
    ds, x, y, data, profile = get_tiff_ds(fire, tiff)

    fields = [
        "Humidity",
        "Precipitation",
        "U10",
        "V10",
        "T2"
    ]

    bands = []
    template = rioxarray.open_rasterio(tiff)

    for field in fields:
        var = ds[field].isel(Time=0).rename({"south_north": "y", "west_east": "x"})
        var = var.assign_coords({"x": x, "y": y})
        var.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
        var.rio.write_crs("EPSG:4326", inplace=True)
        var_reproj = var.rio.reproject_match(template)
        bands.append(var_reproj.values)

    wrf_stack = np.stack(bands) 
    new_data = np.concatenate([data, wrf_stack], axis=0)
    profile.update(count=new_data.shape[0])


    output = OUT_PATH / "fire_" + fire.id / f"{tiff.stem}_plus_wrf.tif"

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(new_data)


def main():
    parse_args()
    
    fires = TIFF_PATH.iterdir()
    for fire_dir in fires:
        print(fires)
        fire = Fire.from_filename(fire_dir.name)
        for tiff in fire.tiff_path.iterdir():
            try:
                append_wrf(fire, tiff)
            except Exception as e:
                if not isinstance(e, OSError):
                    traceback.print_exc()
                else:
                    print(e)

if __name__ == "__main__":

    tiff_path = "./data/using/tiffs/fire_20777134/Image_Export_fire_20777134_2017-07-20.tif"
    wrf_path = "./data/using/wrf/fire_20777134/"

    with rasterio.open(tiff_path) as src:
        profile = src.profile.copy()
        data = src.read()

    date = get_tiff_date(tiff_path)[0]

    files = [
        xr.open_dataset(fp)
        for fp in sorted(Path(wrf_path).glob(f"wrfout_d01_{date}_*:00:00"))
    ]

    for ds in files:
        ds["Precipitation"] = ds["RAINNC"] + ds["RAINC"]

    fields = [
        "Precipitation",
        "U10",
        "V10",
        "T2"
    ]

    for ds in files:
        for field in fields:
            new = ds[field].isel(Time=0).assign_coords({
                "west_east": ds["XLONG"].isel(Time=0).isel(south_north=0).data,
                "south_north": ds["XLAT"].isel(Time=0).isel(west_east=0).data
            })
            new.name = field
            new.rio.set_spatial_dims(x_dim="west_east", y_dim="south_north", inplace=True)
            new.rio.write_crs("EPSG:4326", inplace=True)
            new_matched = new.rio.reproject_match(rioxarray.open_rasterio(tiff_path))
            new_data = np.concatenate([data, new_matched.values[np.newaxis, ...]], axis=0)
            profile.update(count=new_data.shape[0])

    with rasterio.open("gee_plus_wrf.tif", "w", **profile) as dst:
        dst.write(new_data)  #type: ignore

# def check_visualsj(
#     tiff_path: str | Path,
#     wrf_path: str | Path,
#     out_dir: str | Path = ".",
#     prefix: str = "raw_view"
# ):
#     """
#     Use this to check if the images match visually. You can combine this with the 
#     javascript snippet in the GEE code editor to check the boxes as well. 
#     """
#     tiff_path = Path(tiff_path)
#     wrf_path = Path(wrf_path)
#     out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     ds = xr.open_dataset(wrf_path)
#     t2 = ds["T2"].isel(Time=0).values  
#     t2 = np.flipud(t2)
#
#     with rasterio.open(tiff_path) as src:
#         tiff_band = src.read(src.count) 
#
#     vmin = float(np.nanmin([t2.min(), tiff_band.min()]))
#     vmax = float(np.nanmax([t2.max(), tiff_band.max()]))
#
#     def save_image(array, out_file):
#         plt.figure(figsize=(6, 6))
#         plt.imshow(array, cmap="gray", vmin=vmin, vmax=vmax)
#         plt.axis("off")
#         plt.savefig(out_file, dpi=300, bbox_inches="tight", pad_inches=0)
#         plt.close()
#
#     save_image(t2, out_dir / f"{prefix}_wrf.png")
#     save_image(tiff_band, out_dir / f"{prefix}_tiff.png")
#
#     print("Saved PNGs:")
#     print(out_dir / f"{prefix}_wrf.png")
#     print(out_dir / f"{prefix}_tiff.png")
#

