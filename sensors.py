#!/usr/bin/env python
# coding: utf-8

# ### Target Variable Evaluation for PM 2.5 Estimation
# 
# Rupert is aiming to predict the air quality impact of fire (broad statement for now). To do this we need some source of truth to evaluate our predictions against and to use as a target variable when we're training our models. The obvious source of these readings are ground sensors. While there is a large dataset of hourly resolution sensor data available from [AirNow](https://aqs.epa.gov/aqsweb/airdata/download_files.html), it's still sparse compared to the overall area of the US. To cope with that we're going to try and use the air quality estimates from global atmospheric reanalysis datasets to fill in the areas missing from sensor datasets. The aim of this notebook is to evaluate several of these and see if they suit our purposes. 
# 
# ### The Datasets!
# Data frequency, start/end dates, reanalyis type, and other general information. 
# 
# #### AirNow
# #### CAMS
# #### MERRA2 
# #### MERRA2R
# #### CONUS
# 
# # NOTES
# I want these sections
# 
# 1) Overall overview of each of the sources over the entire time range
# 2) Overview of each of the sources per year
# 3) First one but only when there are fires
# 4) second one but only when there are fires

# In[1]:


import pandas as pd
from pathlib import Path

data_dir = Path("./air_quality/data/out")
csv_files = list(data_dir.glob("*.csv"))
data = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)


# In[2]:


data
data["Sample Measurement"] = data["Sample Measurement"].where(data["Sample Measurement"] >= 2, 1)


# In[3]:


data.isna().sum().sort_values(ascending=False)


# In[4]:


count = data[["Latitude", "Longitude"]].drop_duplicates().shape[0]
print(f"Number of sensors: {count}")


# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

base_col = "Sample Measurement"
sat_cols = ["CONUS", "CAMS", "MERRA2", "MERRA2R"]
results = {}

for col in sat_cols:
    subset = data[[base_col, col]].dropna()
    subset = subset[
        (subset[base_col].between(0.1, 1000)) &
        (subset[col].between(0.1, 1000))
    ]

    if subset.empty:
        results[col] = {
            "Pearson": np.nan,
            "Log Pearson": np.nan,
            "Spearman": np.nan,
            "RMSE": np.nan,
            "Bias": np.nan,
            "Slope": np.nan,
        }
        continue

    log_subset = np.log(subset)

    x = subset[col].values.reshape(-1, 1)
    y = subset[base_col].values

    model = LinearRegression().fit(x, y)
    slope = model.coef_[0]

    bias = np.mean(y - x.flatten())

    mse = mean_squared_error(y, x.flatten())
    rmse = np.sqrt(mse)

    pearson = subset[base_col].corr(subset[col], method="pearson")
    log_pearson = log_subset[base_col].corr(log_subset[col], method="pearson")
    spearman = spearmanr(subset[base_col], subset[col], nan_policy="omit").correlation

    results[col] = {
        "Pearson": pearson,
        "Log Pearson": log_pearson,
        "Spearman": spearman,
        "RMSE": rmse,
        "Bias": bias,
        "Slope": slope,
    }

correlations = pd.DataFrame(results).T
print(correlations)

