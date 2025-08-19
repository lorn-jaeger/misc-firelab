import rioxarray as xrio
import xarray as xr
ds = xr.open_mfdataset("data/*.nc4").chunk()
ds
pm25 = ds["MERRA2_CNN_Surface_PM25"]
pm25
pm25 = ds["MERRA2_CNN_Surface_PM25"].rechunk()
pm25 = ds["MERRA2_CNN_Surface_PM25"].chunk()
pm25
bad = (pm25 >= 100).sum()
bad
from dask.diagnostics import ProgressBar
with ProgressBar():
    bad = (pm25 >= 100).sum().compute().item()
bad
bad / pm25.size * 100
bad / pm25.size
pm25.rio.crs
pm25.rio.transform()
pm25
ds
xrio.open_rasterio?
dir(bad.rio)
dir(bad.rio)
bad.rio
dir(pm25.rio)
pm25.rio.set_spatial_dims(x='lon', y='lat')
pm25.rio.set_spatial_dims?
pm25.rio.set_spatial_dims(x_dim='lon', y_dim='lat')
pm25.rio.set_spatial_dims(x_dim='lon', y_dim='lat').rio.transform()
pm25 = ds["MERRA2_CNN_Surface_PM25"].chunk().rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.set_crs("EPSG:4326")
pm25 = ds["MERRA2_CNN_Surface_PM25"].chunk().rio.set_spatial_dims(x_dim="lon", y_dim="lat").rio.write_crs("EPSG:4326")
pm25.rio.transform()
pm25.rio.crs
m = pm25.rio.transform()
m
m * (0, 0)
m * (360, 575)
m * (575, 360)
%history -f session_history.py
