import xml.etree.ElementTree as ET
import csv
from datetime import datetime
import os

tree = ET.parse("./out.xml")
root = tree.getroot()

fire_data = []

base_dir = root.find("./directory[@name='.']")
if base_dir is not None:
    for year_dir in base_dir.findall("directory"):
        for fire_dir in year_dir.findall("directory"):
            fire_name = fire_dir.attrib["name"]
            if fire_name.startswith("fire_"):
                fire_id = fire_name.replace("fire_", "")
                dates = []
                for file_elem in fire_dir.findall("file"):
                    fname = file_elem.attrib["name"]
                    if fname.endswith(".tif"):
                        try:
                            date_str = os.path.splitext(fname)[0]
                            date = datetime.strptime(date_str, "%Y-%m-%d").date()
                            dates.append(date)
                        except ValueError:
                            continue
                if dates:
                    fire_data.append({
                        "fire_id": fire_id,
                        "start_date": min(dates),
                        "end_date": max(dates)
                    })

with open("fires.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fire_id", "start_date", "end_date"])
    writer.writeheader()
    writer.writerows(fire_data)

print("fires.csv written with", len(fire_data), "records.")
