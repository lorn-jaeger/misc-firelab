import os
import re
import xarray as xr
import rioxarray
import rasterio
from rasterio.enums import Resampling

fires_root = "data/fires"
merra_root = "data/merra"
date_re = re.compile(r"(\d{4})-(\d{2})-(\d{2})\.tif")

def to_xy(da):
    rename = {}
    if "lon" in da.dims: rename["lon"] = "x"
    if "lat" in da.dims: rename["lat"] = "y"
    da = da.rename(rename)
    da = da.rio.set_spatial_dims(x_dim="x", y_dim="y")
    da = da.rio.write_crs("EPSG:4326", inplace=True)
    return da

for year in sorted(os.listdir(fires_root)):
    year_path = os.path.join(fires_root, year)
    if not os.path.isdir(year_path):
        continue

    for fire_id in sorted(os.listdir(year_path)):
        fire_path = os.path.join(year_path, fire_id)
        if not os.path.isdir(fire_path):
            continue

        print(f"Processing fire: {fire_path}")

        for fname in sorted(os.listdir(fire_path)):
            m = date_re.match(fname)
            if not m:
                continue

            y, mth, d = m.groups()
            date = f"{y}{mth}{d}"
            fire_tif = os.path.join(fire_path, fname)
            merra_nc = os.path.join(merra_root, f"MERRA2_HAQAST_CNN_L4_V1.{date}.nc4")

            if not os.path.exists(merra_nc):
                print(f"  skip {fname} (no merra for {date})")
                continue

            try:
                fire = rioxarray.open_rasterio(fire_tif)
                base_bands = int(fire.sizes["band"])
                target_dtype = fire.dtype
                nodata = fire.rio.nodata

                ds = xr.open_dataset(merra_nc)

                pm25 = to_xy(ds["MERRA2_CNN_Surface_PM25"]).mean(dim="time").squeeze()
                qflag = to_xy(ds["QFLAG"]).squeeze()

                pm25_match = pm25.rio.reproject_match(fire, resampling=Resampling.bilinear).squeeze()
                qflag_match = qflag.rio.reproject_match(fire, resampling=Resampling.nearest).squeeze()

                if set(pm25_match.dims) != {"y", "x"}:
                    pm25_match = pm25_match.transpose("y", "x")
                if set(qflag_match.dims) != {"y", "x"}:
                    qflag_match = qflag_match.transpose("y", "x")

                pm25_match = pm25_match.astype(target_dtype)
                qflag_match = qflag_match.astype(target_dtype)

                pm25_band_idx = base_bands + 1
                qflag_band_idx = base_bands + 2

                pm25_match = pm25_match.expand_dims("band").assign_coords(band=[pm25_band_idx])
                qflag_match = qflag_match.expand_dims("band").assign_coords(band=[qflag_band_idx])

                combined = xr.concat([fire, pm25_match, qflag_match], dim="band")

                if nodata is not None:
                    combined = combined.rio.write_nodata(nodata)

                existing_ln = fire.attrs.get("long_name")
                if isinstance(existing_ln, (list, tuple)) and len(existing_ln) == base_bands:
                    fire_descs = list(existing_ln)
                else:
                    fire_descs = [f"fire_band_{i+1}" for i in range(base_bands)]
                pm25_desc = pm25.attrs.get("long_name") or "PM25_daily_mean"
                qflag_desc = qflag.attrs.get("long_name") or "QFLAG"
                descs = fire_descs + [pm25_desc, qflag_desc]

                cleaned_attrs = {k: v for k, v in fire.attrs.items() if k != "long_name"}
                combined.attrs = cleaned_attrs

                combined.rio.to_raster(fire_tif)

                with rasterio.open(fire_tif, "r+") as dst:
                    for i, desc in enumerate(descs, start=1):
                        try:
                            dst.set_band_description(i, desc)
                        except Exception:
                            pass

                    pm25_tags = {k: str(v) for k, v in pm25.attrs.items()}
                    qflag_tags = {k: str(v) for k, v in qflag.attrs.items()}
                    if pm25_tags:
                        dst.update_tags(pm25_band_idx, **pm25_tags)
                    if qflag_tags:
                        dst.update_tags(qflag_band_idx, **qflag_tags)

                print(f"  updated {fire_tif} → now {combined.sizes['band']} bands")

            except Exception as e:
                print(f"  ERROR {fire_tif}: {e}")
                continue

