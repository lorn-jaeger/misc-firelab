#!/usr/bin/env python
# coding: utf-8

# In[163]:


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import warnings

warnings.filterwarnings("ignore")


data = pd.read_csv('data/out/out2.csv')

data = data[
    (data["sat_pm25"] > 0) &
    (data["sensor_pm25"] > 0)
].copy()


# In[164]:


import numpy as np

min_val, max_val = 0, 100000
data["year"] = pd.to_datetime(data["time"]).dt.year

correlation_by_year = (
    data[
        (data["year"] < 2024) &
        (data["sat_pm25"] > min_val) &
        (data["sat_pm25"] < max_val) &
        (data["sensor_pm25"] > min_val) &
        (data["sensor_pm25"] < max_val)
    ]
    .dropna(subset=["sat_pm25", "sensor_pm25"])
    .assign(
        log_sat=np.log(data["sat_pm25"]),
        log_sensor=np.log(data["sensor_pm25"])
    )
    .groupby("year")
    .apply(lambda g: g["log_sat"].corr(g["log_sensor"]))
    .reset_index(name="log_correlation")
)

print(correlation_by_year)



# In[165]:


import numpy as np
import pandas as pd
from scipy.stats import spearmanr

min_val, max_val = 0, 100000
data["year"] = pd.to_datetime(data["time"]).dt.year

filtered = data[
    (data["year"] < 2024) &
    (data["sat_pm25"] > min_val) & (data["sat_pm25"] < max_val) &
    (data["sensor_pm25"] > min_val) & (data["sensor_pm25"] < max_val)
].dropna(subset=["sat_pm25", "sensor_pm25"]).copy()

filtered["log_sat"] = np.log(filtered["sat_pm25"])
filtered["log_sensor"] = np.log(filtered["sensor_pm25"])

correlation_by_year = (
    filtered
    .groupby("year")
    .apply(lambda g: spearmanr(g["log_sat"], g["log_sensor"]).correlation)
    .reset_index(name="log_spearman_correlation")
)

print(correlation_by_year)


# In[166]:


import pandas as pd
from scipy.stats import spearmanr

min_val, max_val = 0, 100000
data["year"] = pd.to_datetime(data["time"]).dt.year

filtered = data[
    (data["year"] < 2024) &
    (data["sat_pm25"] > min_val) & (data["sat_pm25"] < max_val) &
    (data["sensor_pm25"] > min_val) & (data["sensor_pm25"] < max_val)
].dropna(subset=["sat_pm25", "sensor_pm25"]).copy()

correlation_by_year = (
    filtered
    .groupby("year")
    .apply(lambda g: spearmanr(g["sat_pm25"], g["sensor_pm25"]).correlation)
    .reset_index(name="spearman_correlation")
)

print(correlation_by_year)


# In[167]:


import numpy as np
min = 0
max = 100000
valid = (data["sat_pm25"] > min) & (data["sensor_pm25"] > min).dropna()
valid = (data["sat_pm25"] < max) & (data["sensor_pm25"] < max).dropna()

log_sat = np.log(data.loc[valid, "sat_pm25"])
log_sensor = np.log(data.loc[valid, "sensor_pm25"])

log_correlation = log_sat.corr(log_sensor)
print("Log Correlation:", log_correlation)


# In[168]:


min_val, max_val = 0, 100000
data["year"] = pd.to_datetime(data["time"]).dt.year

# Filter data
filtered = data[
    (data["year"] < 2024) &
    (data["sat_pm25"] > min_val) & (data["sat_pm25"] < max_val) &
    (data["sensor_pm25"] > min_val) & (data["sensor_pm25"] < max_val)
].dropna(subset=["sat_pm25", "sensor_pm25"]).copy()

# Log-transform
filtered["log_sat"] = np.log(filtered["sat_pm25"])
filtered["log_sensor"] = np.log(filtered["sensor_pm25"])

# Compute correlations
spearman_corr = spearmanr(filtered["sat_pm25"], filtered["sensor_pm25"]).correlation
log_spearman_corr = spearmanr(filtered["log_sat"], filtered["log_sensor"]).correlation
pearson_corr = filtered["sat_pm25"].corr(filtered["sensor_pm25"])
log_pearson_corr = filtered["log_sat"].corr(filtered["log_sensor"])

# Print results
print(f"Spearman correlation:       {spearman_corr:.4f}")
print(f"Log-Spearman correlation:   {log_spearman_corr:.4f}")
print(f"Pearson correlation:        {pearson_corr:.4f}")
print(f"Log-Pearson correlation:    {log_pearson_corr:.4f}")


# In[169]:


from pathlib import Path
import yaml
import pandas as pd
import matplotlib.pyplot as plt

fire_dir = Path("data/fires")
fire_names = pd.DataFrame(columns=['names', 'year'])

for fire_path in fire_dir.glob("us_fire_*_1e7.yml"):
    with fire_path.open() as f:
        fire_file = yaml.safe_load(f)
        year = fire_file.get("year")
        fires = {k: v for k, v in fire_file.items() if k not in ("output_bucket", "rectangular_size", "year")}
        for fire_name in fires:
            fire_names.loc[len(fire_names)] = [fire_name, year]


fire_names["year"] = fire_names["year"].astype(int)
fire_names["names"] = fire_names["names"].astype(str)

yaml_fires_per_year = fire_names.groupby("year")["names"].nunique()

data["time"] = pd.to_datetime(data["time"])
data["year"] = data["time"].dt.year
data["fire_name"] = data["fire_name"].astype(str)

data_fires_per_year = data.groupby("year")["fire_name"].nunique()

plt.figure(figsize=(10, 6))
data_fires_per_year.plot(marker='o', label="Fires with Sensor Data")
yaml_fires_per_year.plot(marker='o', label="Total Fires (YAML)")
plt.title("Number of Fires per Year")
plt.xlabel("Year")
plt.ylabel("Number of Fires")
plt.grid(True)
plt.legend()
plt.tight_layout()# Compute correlation

plt.show()

combined = pd.DataFrame({
    "Fires with Data": data_fires_per_year,
    "Total Fires": yaml_fires_per_year
})
combined["Percent with Data"] = (combined["Fires with Data"] / combined["Total Fires"] * 100).round(2)

# Print stats
print("Yearly Fire Coverage Statistics:\n")
print(combined.fillna(0).astype({"Fires with Data": int, "Total Fires": int}))


# In[170]:


data["diff"] = abs(data["sat_pm25"] - data["sensor_pm25"])
data["diff"].hist(bins=50)
plt.title("Absolute Error Distribution")
plt.xlabel("Absolute Error")
plt.show()


# In[171]:


data.plot.scatter(x="sensor_pm25", y="diff", alpha=0.3)
plt.title("Sensor PM2.5 vs Absolute Error")
plt.show()


# In[172]:


data.plot.scatter(x="sat_pm25", y="diff", alpha=0.3)
plt.title("Satellite PM2.5 vs Absolute Error")
plt.show()


# In[173]:


data["bin"] = (data["sensor_pm25"] // 50) * 50

# Correlation per bin
bin_corrs = (
    data.groupby("bin")
    .apply(lambda g: g["sat_pm25"].corr(g["sensor_pm25"]))
    .dropna()
)

# Plot correlation
bin_corrs.plot(marker='o')
plt.title("Correlation per 10-unit Sensor PM2.5 Bin")
plt.xlabel("Sensor PM2.5 Bin (Lower Bound)")
plt.ylabel("Correlation")
plt.grid(True)
plt.show()


# In[174]:


# Count per bin as a distribution
bin_counts = data.groupby("bin").size().sort_index()

# Plot as a line (distribution style)
bin_counts.plot(marker='o', linestyle='-')
plt.title("Distribution of Sensor PM2.5 Readings by 10-unit Bins")
plt.xlabel("Sensor PM2.5 Bin (Lower Bound)")
plt.ylabel("Number of Observations")
plt.grid(True)
plt.show()


# In[175]:


data["time"] = pd.to_datetime(data["time"])

# Extract year from 'time'
data["year"] = data["time"].dt.year

# Count unique fires per year
fires_per_year = data.groupby("year")["fire_name"].nunique()

# Plot
fires_per_year.plot(marker='o')
plt.title("Number of Unique Fires with Data per Year")
plt.xlabel("Year")
plt.ylabel("Number of Fires")
plt.grid(True)
plt.show()


# In[176]:


fire_locs = data.groupby("fire_name")[["lat", "lon"]].first().reset_index()

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    fire_locs,
    geometry=gpd.points_from_xy(fire_locs["lon"], fire_locs["lat"]),
    crs="EPSG:4326"
).to_crs(epsg=3857)

# Plot
ax = gdf.plot(figsize=(10, 8), alpha=0.6, edgecolor='k')
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)
plt.title("Fire Locations with Sensor and Satellite PM2.5 Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()


# In[177]:


(data["sat_pm25"] - data["sensor_pm25"]).hist(bins=50)
plt.title("Histogram of Satellite - Sensor PM2.5")
plt.xlabel("Error (Satellite - Sensor)")
plt.ylabel("Count")
plt.grid(True)
plt.show()


# In[178]:


data["hour"] = pd.to_datetime(data["time"]).dt.hour
data.groupby("hour")["diff"].mean().plot(marker='o')
plt.title("Average Absolute Error by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Mean Absolute Error")
plt.grid(True)
plt.show()


# In[179]:


data.plot.scatter(x="lat", y="diff", alpha=0.3)
plt.title("Latitude vs Absolute Error")
plt.xlabel("Latitude")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.show()
data.plot.scatter(x="lat", y="diff", alpha=0.3)
plt.title("Latitude vs Absolute Error")
plt.xlabel("Latitude")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.show()


# In[180]:


data.plot.scatter(x="lon", y="diff", alpha=0.3)
plt.title("Latitude vs Absolute Error")
plt.xlabel("Latitude")
plt.ylabel("Absolute Error")
plt.grid(True)
plt.show()


# In[181]:


import seaborn as sns

mean_error_per_fire = data.groupby("fire_name")["diff"].mean()

# Plot smoothed distribution
sns.kdeplot(mean_error_per_fire, fill=True, linewidth=2)
plt.title("Smoothed Distribution of Average Absolute Error by Fire")
plt.xlabel("Average Absolute Error (PM2.5)")
plt.ylabel("Density")
plt.grid(True)
plt.show()


# In[182]:


avg_pm25_per_fire = data.groupby("fire_name")["sensor_pm25"].mean()

# Plot smoothed distribution
sns.kdeplot(avg_pm25_per_fire, fill=True, linewidth=2)
plt.title("Smoothed Distribution of Average Sensor PM2.5 by Fire")
plt.xlabel("Average Sensor PM2.5 (μg/m³)")
plt.ylabel("Density")
plt.grid(True)
plt.show()


# In[183]:


avg_pm25_per_fire = data.groupby("fire_name")["sat_pm25"].mean()

# Plot smoothed distribution
sns.kdeplot(avg_pm25_per_fire, fill=True, linewidth=2)
plt.title("Smoothed Distribution of Average Sensor PM2.5 by Fire")
plt.xlabel("Average Sensor PM2.5 (μg/m³)")
plt.ylabel("Density")
plt.grid(True)
plt.show()


# In[184]:


high_pm25 = data[data["sensor_pm25"] > 100]

# Compute correlation
corr = high_pm25["sensor_pm25"].corr(high_pm25["sat_pm25"])
print("Correlation", corr)

