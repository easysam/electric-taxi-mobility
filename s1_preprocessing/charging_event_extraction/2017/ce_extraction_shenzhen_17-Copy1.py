#!/usr/bin/env python
# coding: utf-8

# # Charging Event Extraction

# ## Adding Distance to CS into Trajectory

# In[1]:


from dask.distributed import Client, LocalCluster
cluster = LocalCluster()
client = Client(cluster)


# In[2]:


import numpy as np
import pandas as pd
import dask.array as da
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn.neighbors import BallTree
from utils.trajectory import coord2radian
from utils.dask_extension import pd_reset_index


# In[3]:


trajectories = dd.read_parquet('ce_extraction_2017/data/trajectory/statuses_wo_charging_resting', 
                               columns=['latitude', 'longitude']).repartition(npartitions=128)
trajectories = pd_reset_index(trajectories)
cs = pd.read_csv('ce_extraction_2017/data/charging_station/chongdianba_wgs84.csv')


# In[4]:


cs_loc = coord2radian(cs['latitude'], cs['longitude'], lat_first=True)
tree = BallTree(cs_loc, leaf_size=15, metric='haversine')
result = trajectories[['latitude', 'longitude']].map_partitions(
    lambda x: tree.query(coord2radian(x['latitude'], x['longitude'])),
    meta={0: float, 1: int})
trajectories = trajectories.assign(s_dis=result[0]*6371008.8)
trajectories = trajectories.assign(s_idx=result[1])


# In[5]:


trajectories.to_parquet('ce_extraction_2017/data/trajectory/trajectories_w_statuses_cdb')

