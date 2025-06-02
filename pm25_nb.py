#!/usr/bin/env python
# coding: utf-8

# In[62]:


import xarray as xr
import earthaccess
import boto3
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import warnings
from IPython.display import display, Markdown
import numpy as np

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[63]:


# Authenticate using Earthdata Login prerequisite files
auth = earthaccess.login(persist=True)


# From GES-DISC web search (https://disc.gsfc.nasa.gov):
# 
# All data available from 1980 to Present
# 
# Need to Calculate PM2.5 from a formula
# 
# PM2.5 Hourly: C1276812830-GES_DISC, s3://gesdisc-cumulus-prod-protected/MERRA2/M2T1NXAER.5.12.4/
# 
#     - https://disc.gsfc.nasa.gov/datasets/M2T1NXAER_5.12.4/summary
# 
# PM2.5 Monthly : C1276812866-GES_DISC, s3://gesdisc-cumulus-prod-protected/MERRA2_MONTHLY/M2TMNXAER.5.12.4/
# 
#     - https://disc.gsfc.nasa.gov/datasets/M2TMNXAER_5.12.4/summary
# 
# PM2.5 Monthly (Hourly Average): C1276812869-GES_DISC, s3://gesdisc-cumulus-prod-protected/MERRA2_DIURNAL/M2TUNXAER.5.12.4/
# 
#     - https://disc.gsfc.nasa.gov/datasets/M2TUNXAER_5.12.4/summary
# 
# MERRA2_CNN_HAQAST bias corrected global hourly surface total PM2.5 mass concentration, V1 (MERRA2_CNN_HAQAST_PM25): C3094710982-GES_DISC, s3://gesdisc-cumulus-prod-protected/HAQAST/MERRA2_CNN_HAQAST_PM25.1/
# 
#     - https://disc.gsfc.nasa.gov/datasets/MERRA2_CNN_HAQAST_PM25_1/summary
#     - 2000-01-01 to  2024-06-01

# In[64]:


# Search for the granule by DOI
results = earthaccess.search_data(
 concept_id="C3094710982-GES_DISC", temporal=("2023-08-18", "2023-08-18"))

print(results)


# In[65]:


fn = earthaccess.open(results)


# In[66]:


print(xr.backends.list_engines())


# In[67]:


ds = xr.open_mfdataset(fn)


# In[68]:


type(ds)


# In[69]:


ds.attrs


# In[70]:


ds.dims


# In[71]:


ds.values


# In[72]:


ds.variables


# In[73]:


ds.MERRA2_CNN_Surface_PM25


# In[76]:


# Subset SLP at 15:30Z, convert units
pm25 = ds.MERRA2_CNN_Surface_PM25[1, :, :].values
pm25


# In[75]:


plt.rcParams['figure.figsize'] = 10,10

# Set up figure
fig = plt.figure()

ax = fig.add_subplot(111, projection=ccrs.LambertConformal())
ax.set_extent([-121, -72, 23, 51], crs=ccrs.PlateCarree()) # CONUS extent
ax.coastlines(resolution='50m')
ax.add_feature(cfeature.BORDERS)
ax.add_feature(cfeature.STATES)

# Set up filled and line contours
filled_c = ax.contourf(ds.lon, ds.lat, pm25, levels=np.linspace(0,100,10), 
                       transform=ccrs.PlateCarree())
line_c = ax.contour(ds.lon, ds.lat, pm25, levels=np.linspace(0,100,10),
                        colors=['black'],
                        transform=ccrs.PlateCarree())

# Lat/lon grid lines
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

# Set up labelling for contours
ax.clabel(line_c,  
    colors=['black'],
    manual=False,  
    inline=True,  
    fmt=' {:.0f} '.format,  
    )

# Set up colorbar and figure title
fig.colorbar(filled_c, orientation='horizontal')
fig.suptitle('MERRA-2 CONUS PM2.5 on 18 August 2023 00:00', fontsize=16)

plt.show()

