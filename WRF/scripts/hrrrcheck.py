
#!/usr/bin/env python3
"""
hrrcheck.py

Checks only the HRRR analysis files (f00) for each hourly cycle on AWS,
falling back to Google Cloud if the file is not found on AWS.

Usage:
python3 check_hrrr_aws_f00.py --start 20250115_00 --end 20250116_12 --native_grid
"""

import argparse, datetime as dt, pandas as pd, requests, logging, sys

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%Y-%m-%dT%H:%M:%S", level=logging.INFO)
log = logging.getLogger(__name__)

AWS_BASE = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
GCP_BASE = "https://storage.googleapis.com/high-resolution-rapid-refresh"
FMT_YMDH = "%Y%m%d_%H"
FMT_YMD = "%Y%m%d"
FMT_HH = "%H"

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--start", required=True, help="first cycle (YYYYMMDD_HH)")
    p.add_argument("--end", required=True, help="last cycle (YYYYMMDD_HH)")
    p.add_argument("--native_grid", action="store_true", help="also check native grid")
    return p.parse_args()

def url_ok(url):
    try:
        return requests.head(url, timeout=10).status_code == 200
    except requests.RequestException:
        return False

def build_urls(cycle_dt, native, base):
    date = cycle_dt.strftime(FMT_YMD)
    hour = cycle_dt.strftime(FMT_HH)
    base_path = f"{base}/hrrr.{date}/conus"
    urls = [f"{base_path}/hrrr.t{hour}z.wrfprsf00.grib2"]
    if native:
        urls.append(f"{base_path}/hrrr.t{hour}z.wrfnatf00.grib2")
    return urls

def main():
    a = parse_args()
    try:
        start = pd.to_datetime(a.start, format=FMT_YMDH)
        end = pd.to_datetime(a.end, format=FMT_YMDH)
    except ValueError:
        log.error("Start/end must be in YYYYMMDD_HH format"); sys.exit(1)
    if end < start:
        log.error("END must be ≥ START"); sys.exit(1)

    cycles = pd.date_range(start, end, freq="1h")
    missing = []

    for cyc in cycles:
        for aws_url, gcp_url in zip(build_urls(cyc, a.native_grid, AWS_BASE),
                                    build_urls(cyc, a.native_grid, GCP_BASE)):
            if url_ok(aws_url):
                log.info(f"OK (AWS) {aws_url}")
            elif url_ok(gcp_url):
                log.info(f"OK (GCP) {gcp_url}")
            else:
                log.warning(f"missing {aws_url} and {gcp_url}")
                missing.append((aws_url, gcp_url))

    log.info(f"Checked {len(cycles)} cycles; missing URLs: {len(missing)}")
    if missing:
        log.info("First 10 missing URL pairs:")
        for aws, gcp in missing[:10]:
            log.info(f"AWS: {aws}")
            log.info(f"GCP: {gcp}")

if __name__ == "__main__":
    main()
