from ee.filter import Filter
from ee.geometry import Geometry
from ee.featurecollection import FeatureCollection
import pandas as pd
import geopandas as gpd
import os
import time
from ee_wildfire.UserInterface import ConsoleUI

usa_coords = [
    [-125.1803892906456, 35.26328285844432],
    [-117.08916345892665, 33.2311514593429],
    [-114.35640058749676, 32.92199940444295],
    [-110.88773544819885, 31.612036247094473],
    [-108.91086200144109, 31.7082477979397],
    [-106.80030780089378, 32.42079476218232],
    [-103.63413436750255, 29.786401496314422],
    [-101.87558377066483, 30.622527701868453],
    [-99.40039768482492, 28.04018292597704],
    [-98.69085295525215, 26.724810345780593],
    [-96.42355704777482, 26.216515704595633],
    [-80.68508661702214, 24.546812350183075],
    [-75.56173032587596, 26.814533788629998],
    [-67.1540159827795, 44.40095539443753],
    [-68.07548734644243, 46.981170472447374],
    [-69.17500995805074, 46.98158998130476],
    [-70.7598785138901, 44.87172183866657],
    [-74.84994741250935, 44.748084983808],
    [-77.62168256782745, 43.005725611950055],
    [-82.45987924104175, 41.41068867019324],
    [-83.38318501671864, 42.09979904377044],
    [-82.5905167831457, 45.06163491639556],
    [-84.83301910769038, 46.83552648258547],
    [-88.26350848510909, 48.143646480291835],
    [-90.06706251069104, 47.553445811024204],
    [-95.03745451438925, 48.9881557770297],
    [-98.45773319567587, 48.94699366043251],
    [-101.7018751401119, 48.98284560308372],
    [-108.43164852530356, 48.81973606668503],
    [-115.07339190755627, 48.93699058308441],
    [-121.82530604190744, 48.9830983403776],
    [-122.22085227110232, 48.63535795404536],
    [-124.59504332589562, 47.695726563030405],
    [-125.1803892906456, 35.26328285844432]
]


def create_usa_geometry():
    return Geometry.Polygon([usa_coords])

def compute_area(feature):
    return feature.set({'area': feature.area()})

def compute_centroid(feature):
    centroid = feature.geometry().centroid().coordinates()
    return feature.set({
        'lon': centroid.get(0),
        'lat': centroid.get(1)
    })

def ee_featurecollection_to_gdf(fc):
    return gpd.GeoDataFrame([f['properties'] for f in fc.getInfo()['features']])

def get_final_fires(config, collection, start, end, region):
    fc = (
            FeatureCollection(collection)
            .filterBounds(region)
            .map(compute_area)
            .filter(Filter.gte('area', config.min_size))
            .filter(Filter.lt('area', 1e20))
            .filter(Filter.gte('IDate', start))
            .filter(Filter.lt('IDate', end))
        )
        
    final = ee_featurecollection_to_gdf(fc)

    return final

def get_daily_fires(region, row):
    years = sorted(set(pd.date_range(
        start=row['IDate'],
        end=row['FDate'],
        freq='D'
    ).year))

    daily = []

    for year in years:
        collection = f"JRC/GWIS/GlobFire/v2/DailyPerimeters/{year}"
        fc = (
            FeatureCollection(collection)
            .filterBounds(region)
            .filter(Filter.eq('Id', row['Id']))
            .map(compute_centroid)
        )

        gdf = ee_featurecollection_to_gdf(fc)

        if not gdf.empty:
            daily.append(gdf)

    if not daily:
        return gpd.GeoDataFrame(columns=['Id', 'IDate', 'lat', 'lon']) # type: ignore

    daily = gpd.GeoDataFrame(pd.concat(daily, ignore_index=True))

    return daily


def get_initial_coordinates(row, region):

    gdf = get_daily_fires(region, row)
    
    if gdf.empty:
        return pd.Series({'lat': pd.NA, 'lon': pd.NA})

    gdf['IDate'] = pd.to_datetime(gdf['IDate'], unit='ms')
    gdf['timedelta'] = (row['IDate'] - gdf['IDate']).abs()
    gdf = gdf.sort_values('timedelta')
    
    if gdf['timedelta'].iloc[0] > pd.Timedelta(hours=24):
            return pd.Series({'lat': pd.NA, 'lon': pd.NA})

    return pd.Series({'lat': gdf['lat'].iloc[0], 'lon': gdf['lon'].iloc[0]})


def format_gdf(fires):
    fires = fires.dropna(subset=['Id', 'IDate', 'FDate'])
    fires['IDate'] = pd.to_datetime(fires['IDate'], unit='ms')
    fires['FDate'] = pd.to_datetime(fires['FDate'], unit='ms')

    fires['lat'] = pd.NA
    fires['lon'] = pd.NA

    fires = fires[['Id', 'IDate', 'FDate', 'lat', 'lon', 'area']]
 
    return fires



import pandas as pd
import numpy as np
from pprint import pprint

def get_fires(config):
    collection = 'USFS/GTAC/MTBS/burned_area_boundaries/v1'
    region = create_usa_geometry()

    start = pd.to_datetime("2016-01-01").value // 10**6
    end = pd.to_datetime("2021-12-31").value // 10**6

    fc = (
        FeatureCollection(collection)
        .filterBounds(region)
        .filter(Filter.gte('Ig_Date', start))
        .filter(Filter.lt('Ig_Date', end))
        .map(compute_area)
        .filter(Filter.gte('area', 10_000_000))
    )

    areas = fc.aggregate_array('area').getInfo()

    print(len(areas))





    


def get_fires_glob(config):
    collection = 'JRC/GWIS/GlobFire/v2/FinalPerimeters'
    region = create_usa_geometry()
    gdfs = []
    weeks = pd.date_range(start=config.start_date, end=config.end_date, freq='W')

    ConsoleUI.add_bar("fires", total=len(weeks), desc=collection, color="green")


    for week in weeks:
        start = int(week.timestamp() * 1000)
        end = int((week + pd.Timedelta(days=1)).timestamp() * 1000)
        gdf = get_final_fires(config, collection, start, end, region)
        if not gdf.empty:
            gdfs.append(gdf)
        ConsoleUI.update_bar("fires")

    fires = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    fires = format_gdf(fires)

    # Applying to the whole dataframe. This is iterating over rows
    # internally and takes about as long as the uncommented way
    
    # tqdm.pandas(desc="Initial coordinates")
    # fires[['lat', 'lon']] = fires.progress_apply(
    #     get_initial_coordinates,
    #     axis=1,
    #     args=(region,),
    # )
    #
    # fires = fires.dropna()
    #
    # return fires
    #

    ConsoleUI.add_bar("coords", total=len(fires), desc="Initial coordinates", color="green")
    lats = []
    lons = []

    for _, row in fires.iterrows():
        lat, lon = get_initial_coordinates(row, region)
        lats.append(lat)
        lons.append(lon)
        ConsoleUI.update_bar("coords")


    fires['lat'] = lats
    fires['lon'] = lons
    fires = fires.dropna()    

    return fires



def sanitize_filename(value):
    return str(value).replace(":", "-").replace(" ", "_")

def get_fire_cache_path(config):
    output_dir = os.path.join(config.data_dir, "gdfs")
    os.makedirs(output_dir, exist_ok=True)

    start = sanitize_filename(config.start_date)
    end = sanitize_filename(config.end_date)
    filename = f"{start}_{end}_{config.min_size}_FIRES.pkl"

    return os.path.join(output_dir, filename)


def save_fires(config):
    ConsoleUI.print("Caching fire query...")
    time.sleep(1) # can remove these. this is just so the message shows
    output_path = get_fire_cache_path(config)
    config.geodataframe.to_pickle(output_path)

def load_fires(config):
    input_path = get_fire_cache_path(config)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Cached fire data not found: {input_path}")

    ConsoleUI.print("GeoDataFrame already exists. Loading from file cache instead.")
    time.sleep(1)
    config.geodataframe = pd.read_pickle(input_path)
