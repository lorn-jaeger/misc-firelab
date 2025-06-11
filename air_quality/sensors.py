from pathlib import Path
import pandas as pd
import argparse
import warnings
from ee.geometry import Geometry
from ee.imagecollection import ImageCollection

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import geemap

SENSOR_PATH = Path("data/sensors")
TEST_PATH = Path("data/test/sensors")
OUT_PATH = Path("data/out")

def clear_output():
    """
    Clear the csv output.
    """
    for file in OUT_PATH.iterdir():
        if file.is_file():
            file.unlink()


def parse_args():
    """
    Simple argument parser with some useful commands.
    Allows me to set a test dir and clear the output.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--clear-out", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    if args.clear_out:
        clear_output() 

    global SENSOR_PATH

    if args.test:
        SENSOR_PATH = TEST_PATH

def save(file):
    file.to_csv("data/out/out", index=False)

def try_auth():
    geemap.ee_initialize()


def CONUS(sensors):
    sensors["CONUS"] = pd.NA
    return sensors


def MERRA2(sensors):
    sensors["MERRA2"] = pd.NA
    return sensors


def MERRA2R(sensors):
    sensors["MEERA2R"] = pd.NA
    return sensors

def AIRNOW(sensors):
    sensors["AIRNOW"] = pd.NA
    return sensors


def CAMS(sensors):
    sensors["CAMS"] = pd.NA
    return sensors

def fmt_sensors(sensors):
    sensors['Date GMT'] = pd.to_datetime(sensors['Date GMT'])
    sensors['Time'] = sensors['Date GMT'] + pd.to_timedelta(sensors['Time GMT'] + ':00')
    
    sensors = sensors[["Time", "Latitude", "Longitude", "Sample Measurement"]]
    return sensors



def main() -> None:
    parse_args()
    try_auth()

    files = SENSOR_PATH.iterdir()
    sources = [CAMS]

    for file in files:
        print(f"Reading {file.name}")
        sensors = pd.read_csv(file, low_memory=False)
        for source in sources:
            sensors = fmt_sensors(sensors)
            sensors = source(sensors)
            print(sensors)
        save(sensors)

if __name__ == "__main__":
    main()



