#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

frm = pd.read_csv("./air_quality/data/out/out_hourly_88101_2016.csv")
non_frm = pd.read_csv("./air_quality/data/out/out_hourly_88502_2016.csv")
data = pd.concat([frm, non_frm], ignore_index=True)



# In[2]:


data
data["Sample Measurement"] = data["Sample Measurement"].where(data["Sample Measurement"] >= 2, 1)


# In[3]:


data.isna().sum().sort_values(ascending=False)


# In[4]:


count = data[["Latitude", "Longitude"]].drop_duplicates().shape[0]
print(f"Number of sensors: {count}")


# In[5]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

base_col = "Sample Measurement"
sat_cols = ["CONUS"]
results = {}

for col in sat_cols:
    subset = data[[base_col, col]].dropna()
    subset = subset[
        (subset[base_col].between(1, 1000)) &
        (subset[col].between(1, 1000))
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


# In[6]:


data["CONUS"].min()


# In[7]:


data["CONUS"].min()


# In[8]:


print(data["Sample Measurement"].max())


# In[9]:


max_idx = data["CONUS"].idxmin()
value = data.loc[max_idx, "Sample Measurement"]
print(value)


# In[10]:


num_under_1 = data["CONUS"].dropna().lt(1).sum()
print(num_under_1)


# In[ ]:




