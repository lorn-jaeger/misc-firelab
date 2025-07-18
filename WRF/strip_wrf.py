import xarray as xr
import argparse
from pathlib import Path
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wrf-path', type=Path, required=True)
    parser.add_argument('--out-path', type=Path, required=True)
    args = parser.parse_args()

    return args

def get_directory_size(wrf_path):
    result = subprocess.run(['du', '-sh', wrf_path], stdout=subprocess.PIPE, text=True)
    return result.stdout.split()[0]

def strip_wrfout(file):
    ds = xr.open_dataset(file)

    coords = ds.coords
    attrs = ds.attrs

    ds["Total Precipitation"] = ds["RAINNC"] + ds["RAINC"] 
    ds["Specific Humidity"] = ds["Q2"] / (1 + ds["Q2"])

    fields = [
        "Total Precipitation",
        "Specific Humidity",
        "V10",
        "U10",
        "T2",
        "XLAT",
        "XLONG",
    ]

    ds = (
            ds[fields]
            .assign_coords(coords)
            .assign_attrs(attrs)
    )

    return ds

def write_file(file,  ds, out_path):
    parent = file.parent.name
    target = out_path / parent

    target.mkdir(parents=True, exist_ok=True)

    out_file = target / file.name
    ds.to_netcdf(out_file, format="NETCDF4_CLASSIC")
    
def main():
    args = parse_args()
    wrf_path = args.wrf_path
    out_path = args.out_path

    print(f"Starting Size: {get_directory_size(wrf_path)}") 

    files = wrf_path.iterdir()
    for file in files:
        try:
            ds = strip_wrfout(file)
            write_file(file, ds, out_path)
            print(f"Stripped {file.name}")
        except Exception as e:
            print(f"Error reading {file.name}: {e}")

    print(f"Ending Size: {get_directory_size(out_path)}") 

if __name__ == '__main__':
    main()
        




   




