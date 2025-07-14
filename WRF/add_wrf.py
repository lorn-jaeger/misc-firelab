import rasterio
from rasterio.transform import xy
from rasterio.warp import transform
from pathlib import Path

tiffs = Path("./data/using/tiffs/")
wrf = Path("./data/using/wrf/")
out = Path("./data/using/out/")



def main():
    with rasterio.open("./data/using/tiffs/Image_Export_fire_20777134_2017-07-15.tif") as src:
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

        center_row = height // 2
        center_col = width // 2

        utm_lon, utm_lat = xy(src.transform, center_row, center_col)

        lon, lat = transform(   #type: ignore
            src.crs,            
            "EPSG:4326",         
            [utm_lon], [utm_lat] 
        )

        print(lat, lon)

if __name__ == '__main__':
    main()




