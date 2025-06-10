from pathlib import Path
import pandas as pd
import argparse

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

def CAMS(file):
    return file

def CONUS(file):
    return file

def MERRA2(file):
    return file

def MERRA2R(file):
    return file

def main() -> None:
    parse_args()

    for file in SENSOR_PATH.iterdir():
        print(f"Reading {file.name}")
        file = pd.read_csv(file, low_memory=False)
        for source in [CAMS, CONUS, MERRA2, MERRA2R]:
            file = source(file)
        



        






if __name__ == "__main__":
    main()



