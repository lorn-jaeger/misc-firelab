import ee, pandas as pd, pathlib as pl
from tqdm import tqdm
ee.Initialize(project='ee-earthdata-459819')

def fmt_sensors(df: pd.DataFrame) -> pd.DataFrame:
    # ‘Date GMT’ is a date; ‘Time GMT’ is an integer hour (0-23)
    df = df.copy()
    df['Date GMT'] = pd.to_datetime(df['Date GMT'])
    df['Time'] = df['Date GMT'] + pd.to_timedelta(df['Time GMT'] + ":00")
    return df[['Time', 'Latitude', 'Longitude', 'Sample Measurement']]

def get_unique(df: pd.DataFrame) -> pd.DataFrame:
    out = (
        df.groupby(['Latitude', 'Longitude'])['Time']
          .agg(Start_Time='min', End_Time='max')
          .reset_index()
    )
    return out

def cams_sample(csv_path: pl.Path) -> str:
    print(f'→ {csv_path.name}')
    sensors = pd.read_csv(csv_path, low_memory=False)
    sensors = fmt_sensors(sensors)
    ranges  = get_unique(sensors)

    stations = ee.FeatureCollection([
        ee.Feature(
            ee.Geometry.Point(r['Longitude'], r['Latitude']),
            {'lon': r['Longitude'], 'lat': r['Latitude']}
        ) for _, r in ranges.iterrows()
    ])

    start = ee.Date(ranges['Start_Time'].min())
    end   = ee.Date(ranges['End_Time'].max()).advance(1, 'day')

    imgs = (ee.ImageCollection('ECMWF/CAMS/NRT')
              .select('particulate_matter_d_less_than_25_um_surface')
              .filter('model_initialization_hour == 0')
              .filterDate(start, end))

    def sample(img):
        t = img.date().format('YYYY-MM-dd HH:mm')
        return img.reduceRegions(
            collection = stations,
            reducer    = ee.Reducer.first(),
            scale      = 10_000
        ).map(lambda f: f.set('time', t))

    table = imgs.map(sample).flatten()

    
    task = ee.batch.Export.table.toDrive(
        collection     = table,
        description    = f'cams_{csv_path.stem}',
        folder         = 'EarthEngine',         # Drive folder (creates it if needed)
        fileNamePrefix = f'cams_{csv_path.stem}',
        fileFormat     = 'CSV'
    )
    task.start()
    return task.id

def main():
    tasks = [cams_sample(f) for f in pl.Path('data/sensors').glob('*.csv')]
    print(f'Submitted {len(tasks)} export tasks.')

if __name__ == '__main__':
    main()

