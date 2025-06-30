from conus_download import downloadFile, processArguments, Download, executeDownload, data
import xarray as xr
import datetime
from pyproj import Proj, Transformer
from pathlib import Path



def check():

    ncdf_path = Path("./PM25_ONLY.nc")
    
    ds = xr.open_dataset(ncdf_path, decode_cf=False)  

    lat, lon = 46.8721, -113.9940

    # Set up Lambert Conformal projection based on dataset attributes
    proj = Proj(proj='lcc',
                lat_1=ds.attrs['P_ALP'],
                lat_2=ds.attrs['P_BET'],
                lat_0=ds.attrs['YCENT'],
                lon_0=ds.attrs['XCENT'],
                x_0=0, y_0=0, ellps='sphere')

    transformer = Transformer.from_proj("epsg:4326", proj, always_xy=True)

    # Convert lon/lat to x/y (in meters)
    x, y = transformer.transform(lon, lat)

    # Convert x/y to COL/ROW
    XORIG = ds.attrs['XORIG']
    YORIG = ds.attrs['YORIG']
    XCELL = ds.attrs['XCELL']
    YCELL = ds.attrs['YCELL']

    col = int((x - XORIG) / XCELL)
    row = int((y - YORIG) / YCELL)

    # Choose a timestep (e.g., 0 = start of simulation)
    t = 0

    # Extract PM2.5 value
    pm25 = ds['PM25_TOT'].isel(TSTEP=t, LAY=0, ROW=row, COL=col).values.item()

    # Optionally get the timestamp
    base_date = datetime.datetime.strptime(str(ds.attrs['SDATE']), "%Y%j")
    step_hours = int(str(ds.attrs['TSTEP'])[:2])  # assuming TSTEP = 10000 → 1hr
    timestamp = base_date + datetime.timedelta(hours=step_hours * t)

    print(f"PM2.5 at Missoula (lat={lat}, lon={lon}) on {timestamp} = {pm25} µg/m³")

def strip_conus(nc_path: Path, out_path: Path):
    """
    Opens a NetCDF file, strips all variables except PM25_TOT,
    and copies only relevant projection-related attributes.
    """
    ds = xr.open_dataset(nc_path, decode_cf=False)
    ds_pm25 = ds[["PM25_TOT"]]

    projection_attrs = ['XORIG', 'XCELL', 'YORIG', 'YCELL',
                        'P_ALP', 'P_BET', 'P_GAM', 'XCENT', 'YCENT', 'SDATE', 'TSTEP']
    ds_pm25.attrs = {k: ds.attrs[k] for k in projection_attrs if k in ds.attrs}
    ds_pm25.to_netcdf(out_path)
    ds.close()
    print(f"Stripped and saved: {out_path}")


def process_all(data_entries):
    args = processArguments()

    for datum in data_entries:
        download = Download(args, datum)
        print(f"\n→ Processing {download.filename}")

        executeDownload(download)

        if download.success and (download.valid or download.vwarning):
            original_path = Path(download.filename)
            stripped_path = original_path.with_stem(original_path.stem + "_PM25")

            # Strip and save
            strip_conus(original_path, stripped_path)

            # Delete the original to free space
            original_path.unlink()
            print(f"Deleted original file: {original_path}")

# Call it with your data list
if __name__ == "__main__":
    process_all(data)


