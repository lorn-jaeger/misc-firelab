import rasterio


with rasterio.open("./data/tiffs/Image_Export_fire_20229545_2016-12-28.tif") as src:
    print("Number of bands:", src.count)
    print("Width, Height:", src.width, src.height)
    print("CRS (projection):", src.crs)
    print("Transform (affine):", src.transform)
    print("Resolution:", src.res)  # (x_res, y_res)
    print("Data type:", src.dtypes)
    print("Bounds:", src.bounds)
    print("Driver:", src.driver)

    #:CEN_LAT = 41.53681f ;
	#:CEN_LON = -123.5552f ;
