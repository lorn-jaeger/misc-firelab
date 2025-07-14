import rasterio
import xarray as xr
import rioxarray
import numpy as np
from pathlib import Path
import argparse
from dataclasses import dataclass
import datetime


WRF_PATH = Path("data/wrf")
TIFF_PATH = Path("data/tiff")
OUT_PATH = Path("data/out")

@dataclass
class Fire:
    id: int
    date: datetime.date
    path: Path
    
    @classmethod
    def from_filename(cls, filename):
        path = TIFF_PATH / filename
        stem = Path(filename).stem
        parts = stem.split('_')

        if len(parts) != 5 or parts[2] != "fire":
            raise ValueError(f"Check file format: {filename}")

        id = int(parts[3])
        date = datetime.date.fromisoformat(parts[4])

        return cls(id=id, date=date, path=path)
        
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

def append_wrf(fire):
    pass

def main():
    parse_args()

    fires = TIFF_PATH.iterdir()

    for fire_dir in fires:
        fire = Fire.from_filename(fire_dir.name)
        append_wrf(fire)



if __name__ == "__main__":
    pass

    #
    # tiff_path = "./data/using/tiffs/Image_Export_fire_20777134_2017-07-15.tif"
    # wrf_path = "./data/using/wrf/wrfout_d01_2017-07-15_00:00:00"
    #
    # with rasterio.open(tiff_path) as src:
    #     profile = src.profile.copy()
    #     data = src.read()
    #
    # ds = xr.open_dataset(wrf_path)
    # t2 = ds["T2"].isel(Time=0)
    # lats = ds["XLAT"].isel(Time=0)
    # lons = ds["XLONG"].isel(Time=0)
    #
    # t2 = t2.rename({"south_north": "y", "west_east": "x"})
    # lats = lats.rename({"south_north": "y", "west_east": "x"})
    # lons = lons.rename({"south_north": "y", "west_east": "x"})
    #
    # x = lons.isel(y=0).data
    # y = lats.isel(x=0).data
    # t2 = t2.assign_coords({"x": x, "y": y})
    #
    # t2.name = "T2_Celsius"
    # t2.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=True)
    # t2.rio.write_crs("EPSG:4326", inplace=True) 
    #
    # template = rioxarray.open_rasterio(tiff_path)
    # t2_matched = t2.rio.reproject_match(template)
    #
    # t2_band = t2_matched.values[np.newaxis, ...] 
    # new_data = np.concatenate([data, t2_band], axis=0)
    # profile.update(count=new_data.shape[0])
    #
    # with rasterio.open("gee_plus_wrf.tif", "w", **profile) as dst:
    #     dst.write(new_data)
    #
    #
    #



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

