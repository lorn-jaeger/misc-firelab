import xarray as xr

ds = xr.open_mfdataset("./data/*.nc4").load()

pm25 = ds["MERRA2_CNN_Surface_PM25"]

bad = (pm25 >= 100).sum().compute().item()
count =pm25.size

percent = (bad / count) * 100

print(percent)




