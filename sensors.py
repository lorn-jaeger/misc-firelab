#!/usr/bin/env python
# coding: utf-8

# 
# ### Target Variable Evaluation for PM 2.5 Estimation
#  
# Rupert is aiming to predict the air quality impact of fire (broad statement for now). To do this we need some source of truth to evaluate our predictions against and to use as a target variable when we're training our models. The obvious source of these readings are ground sensors. While there is a large dataset of hourly resolution sensor data available from [AirNow](https://aqs.epa.gov/aqsweb/airdata/download_files.html), it's still sparse compared to the overall area of the US. To cope with that we're going to try and use the air quality estimates from global atmospheric reanalysis datasets to fill in the areas missing from sensor datasets. The aim of this notebook is to evaluate several of these and see if they suit our purposes. 
# 
# ### The Datasets
# 
# First, a short description of each dataset.
# 
# #### CAMS
# 
# CAMS is a global atmospheric aerosol assimilation model developed by ECMWF (Europeans). The dataset spans from 2016 to 2025. The temporal resolution of the dataset is hourly, but only two readings per day are actual measurements (when a sattelite passes overhead every 12 hours), the rest are forecasts from those measuremts. The spatial resolution of the dataset is $44 km^2$. 
# 
# #### NCAR
# 
# This is NCAR's air quality reanalysis. They take aerosol optical depth (a measure of how much the sun is blocked/scattered by the atmosphere) from MODIS and assimilate that with CMAQ (an atmospheric chemistry tranport model driven by outputs from WRF). The dataset has an hourly temporal resolution and $12 km^2$ spatial resolution. The dataset spans from 2005 to 2018. 
# 
# #### MERRA2
# 
# MERRA2 is NASA's global atmospheric aeresol reanalysis dataset. They take atmospheric data from GEOS (NASA's atmospheric model) and do a bunch of real data assimilation on top of it. The dataset is available from 1980 to present, hourly, at a spatial resolution of $44 km^2$ (maybe not square pixels? I have to check). 
# 
# #### MERRA2R
# 
# MERRA2R is a dataset from NASA that attempts to improve MERRA2's PM 2.5 estimation by using a model trained on ground sensor data and MERRA2 output, which is then applied to correct the original MERRA2 PM 2.5 values. The data has the same resolution as MERRA2 but is only available from 2000 to 2024. 
# 
# #### MERRA2R - 2018
# 
# MERRA2R is only trained on the year 2018 and seems to do somewhat better at predicting air quality in that year. Because we are concerned that MERRA2R is overfit we'll also evalate MERRA2R without that year. 

# ### Sensor Dataset
# 
# I've compiled sensor readings from AirNow with readings from each of these models at the sensors site for the past 20 years. This works out to about 130 millions readings from 1500ish sensors. For most of the datasets I only pulled data from years that we have fire data for as that is what we are interested in anyway. 

# In[1]:


import pandas as pd
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

data_dir = Path("./air_quality/data/final")
csv_files = list(data_dir.glob("*.csv"))
data = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# In[2]:


data["Time"] = pd.to_datetime(data["Time"])
data["MERRA2R_trimmed"] = data.loc[data["Time"].dt.year != 2018, "MERRA2R"]


# In[3]:


data


# In[4]:


count = data[["Latitude", "Longitude"]].drop_duplicates().shape[0]
print(f"Number of sensors: {count}")


# ### Overall Metrics
# 
# These are metrics over every reading available over the entire time period that they are available. 

# In[5]:


from helpers import correlations

correlations(data)


# In[6]:


from helpers import yearly_correlations

yearly_correlations(data, "CAMS")


# In[7]:


yearly_correlations(data, "CONUS")


# In[8]:


yearly_correlations(data, "MERRA2")


# In[9]:


yearly_correlations(data, "MERRA2R")


# In[10]:


fires = pd.read_csv("fires2.csv")


# In[11]:


count = fires[["Latitude", "Longitude"]].drop_duplicates().shape[0]
print(f"Number of sensors: {count}")


# In[12]:


fires["Time"] = pd.to_datetime(fires["Time"])
fires["MERRA2R_trimmed"] = fires.loc[fires["Time"].dt.year != 2018, "MERRA2R"]

correlations(fires)


# In[13]:


yearly_correlations(fires, "CAMS")


# In[14]:


yearly_correlations(fires, "CONUS")


# In[15]:


yearly_correlations(fires, "MERRA2")


# In[16]:


yearly_correlations(fires, "MERRA2R")


# In[17]:


fires.to_csv("air_quality/data/fires.csv")

