from pyproj import Proj, Transformer
from datetime import datetime, timedelta
import numpy as np
import xarray as xr 
from pathlib import Path

def convert(lat, lon, ds):
    proj = Proj(
        proj="lcc",
        lat_1=ds.attrs["P_ALP"],
        lat_2=ds.attrs["P_BET"],
        lat_0=ds.attrs["YCENT"],
        lon_0=ds.attrs["XCENT"],
        x_0=0,
        y_0=0,
        ellps='sphere'
    )

    transformer = Transformer.from_proj("epsg:4326", proj, always_xy=True)

    x, y = transformer.transform(lon, lat)

    col = int((x - ds.XORIG) / ds.XCELL)
    row = int((y - ds.YORIG) / ds.YCELL)

    return row, col


if __name__ == "__main__":

    path = Path("./data/conus/")


    directory = Path("./data/conus")
    files_2016 = sorted(str(f) for f in directory.glob("*2016*.nc"))
    

    ds = xr.open_mfdataset(
        files_2016,
        combine="nested",    
        concat_dim="TSTEP", 
        decode_cf=False     
    )

    sdate = ds.attrs["SDATE"]  
    tstep = ds.attrs["TSTEP"] 

    print(sdate, tstep)

    year = sdate // 1000
    doy = sdate % 1000

    start_time = datetime(year, 1, 1) + timedelta(days=int(doy) - 1)

    nt = ds.sizes["TSTEP"]
    datetimes = [start_time + timedelta(hours=i) for i in range(nt)]

    ds = ds.assign_coords(time=("TSTEP", datetimes))

    print(ds)


    row, col = convert(40.0, -97.0, ds)
    print(f"Row: {row}, Col: {col}")


    pm25_series = ds["PM25_TOT"].isel(ROW=row, COL=col, LAY=0).load().values
    times = ds["time"].values


    times = ds["time"].values 

    import pandas as pd

    df = pd.DataFrame({
        "time": times,
        "PM25": pm25_series
    })

    print(df.head())
    print(df.shape)
    print(df)



















    


    









