#!/usr/bin/env python
# coding: utf-8

# 
# ### Target Variable Evaluation for PM 2.5 Estimation
#  
# Rupert is aiming to predict the air quality impact of fire (broad statement for now). To do this we need some source of truth to evaluate our predictions against and to use as a target variable when we're training our models. The obvious source of these readings are ground sensors. While there is a large dataset of hourly resolution sensor data available from [AirNow](https://aqs.epa.gov/aqsweb/airdata/download_files.html), it's still sparse compared to the overall area of the US. To cope with that we're going to try and use the air quality estimates from global atmospheric reanalysis datasets to fill in the areas missing from sensor datasets. The aim of this notebook is to evaluate several of these and see if they suit our purposes. 
# 
# 
# ##### TO-DO
# - Add my sanity check code that reproduces the metrics from each paper for each dataset
# - Short description of each dataset
# - Some plots instead of tables?
# - Whatever rupurt wants

# In[1]:


import pandas as pd
from pathlib import Path

data_dir = Path("./air_quality/data/out")
csv_files = list(data_dir.glob("*.csv"))
data = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# In[2]:


data.isna().sum().sort_values(ascending=False)


# In[3]:


data.shape


# In[4]:


data


# In[5]:


count = data[["Latitude", "Longitude"]].drop_duplicates().shape[0]
print(f"Number of sensors: {count}")


# In[6]:


from helpers import correlations

correlations(data)


# In[7]:


from helpers import yearly_correlations

yearly_correlations(data, "CAMS")


# In[8]:


yearly_correlations(data, "CONUS")


# In[9]:


yearly_correlations(data, "MERRA2")


# In[10]:


data = (
    data[data["Sample Measurement"] > 1]
)

yearly_correlations(data, "MERRA2R")


# In[11]:


fires = pd.read_pickle("air_quality/data/fires.pkl")


# In[ ]:


fires.shape


# In[ ]:


fires


# In[12]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import box

data["Time"] = pd.to_datetime(data["Time"])
fires["IDate"] = pd.to_datetime(fires["IDate"])
fires["FDate"] = pd.to_datetime(fires["FDate"])

fires["geometry"] = fires.apply(
    lambda row: box(row["lon"] - 1, row["lat"] - 1, row["lon"] + 1, row["lat"] + 1),
    axis=1
)
fires_gdf = gpd.GeoDataFrame(fires, geometry="geometry", crs="EPSG:4326")

min_date = fires["IDate"].min()
max_date = fires["FDate"].max()

data_subset = data[(data["Time"] >= min_date) & (data["Time"] <= max_date)]

data_subset["geometry"] = gpd.points_from_xy(data_subset["Longitude"], data_subset["Latitude"])
data_gdf = gpd.GeoDataFrame(data_subset, geometry="geometry", crs="EPSG:4326")

joined = gpd.sjoin(data_gdf, fires_gdf, predicate="within", how="inner")

filtered = joined[(joined["Time"] >= joined["IDate"]) & (joined["Time"] <= joined["FDate"])]

fires = filtered[data.columns]


# In[13]:


count = fires[["Latitude", "Longitude"]].drop_duplicates().shape[0]
print(f"Number of sensors: {count}")


# In[14]:


correlations(fires)


# In[15]:


import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

yearly_correlations(fires, "CAMS")


# In[16]:


yearly_correlations(fires, "CONUS")


# In[17]:


yearly_correlations(fires, "MERRA2")


# In[ ]:


yearly_correlations(fires, "MERRA2R")


# In[19]:


fires.to_csv("air_quality/data/fires.csv")

