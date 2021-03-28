#!/usr/bin/env python
# coding: utf-8

# # 将原始csv文件转为hdf文件，提高性能

# In[1]:


import numpy as np
import pandas as pd
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


# In[2]:


# 字段说明：车牌-时间-经纬度-速度-方向-载客-公司-颜色-车型（比亚迪e6是电车，其它为非电车）
dtypes = { 0: str, 1: str, 2: float, 3: float, 4: np.int16, 5: np.int16, 
          6: np.int16, 7: str, 8: str, 9: str, 10: str, 11: str}
column_list = ['plate', 'timestamp', 'lng', 'lat', 'speed', 'direction', 
               'passenger', 'company', 'color', 'prototype', 'category']
column_name = {i: item for i, item in enumerate(column_list)}
raw = dd.read_csv(['dataset/20180724to30.gz/part-r-00000', 
                   'dataset/20180724to30.gz/part-r-00001', 
                   'dataset/20180724to30.gz/part-r-00002'], 
                  header=None, dtype=dtypes)
raw = dd.read_csv('dataset/20180724to30.gz/*', header=None, dtype=dtypes)
raw_renamed = raw.rename(columns=column_name)


# ### scheduler的对比（threads vs processes）
# 1. 读3个文件时，使用默认调度器(thread)：1m3s；使用processes：47.6s；使用multiprocessing：46.9s。<br>
# 2. 读62个文件时，使用默认调度器(thread)：21m58.4s；使用processes：16m16.8s。<strong>processes is 1.35x fast than thrad</strong><br>
# The comparison is using the <strong>csv</strong> format file, and in common we will instead using the high performance parquet format file.

# In[ ]:


# To see whether there existing value missing.
with ProgressBar():
    missing_values = raw_renamed.isnull().sum()
    percent_missing = ((missing_values / raw_renamed.index.size) * 100).compute(scheduler='threads')


# ### <strong>hdf</strong> vs <strong>parquet</strong> in writing performance
# hdf need 10m53s<br>
# parquet need 25m5.4s

# In[ ]:


# Convert the origin file to hdf format, to imporove the IO performance. Consuming 10m53.1s
with ProgressBar():
    raw_renamed.to_hdf('dataset/raw_et_shenzhen_trajectory/*.hdf', 'trajectory', 
                       data_columns=column_list, min_itemsize={'color': 15, 'company': 48, 'prototype': 18}, 
                       scheduler='processes')


# In[5]:


# Convert the origin file to parquet format, to imporove the IO performance.
with ProgressBar():
    raw_renamed.to_parquet('dataset/raw_trajectory_et_sz_20180724to30.pqt')


# # 读columns子集、转换时间格式、转换车型编码、排序

# ### <strong>hdf</strong> vs <strong>parquet</strong> in reading performance
# hdf need 9m21s for 9% loading<br>
# parquet need 9m39s for 100% loading
# ### columns selection <strong>in read_parquet</strong> vs <strong>by column selector</strong>
# in read_parquet: 29.8s better!<br>
# by column selector: 1m13.2s
# ### unique value of a column using <strong>threads</strong> vs <strong>processes</strong>
# processes: 32 + 7s better!<br>
# threads: 0 + 1m17s

# In[7]:


import dask.dataframe as dd
from utils.dask_extension import pd_reset_index
raw_pqt = dd.read_parquet('dataset/raw_trajectory_et_sz_20180724to30.pqt', columns=['plate', 'timestamp', 'lng', 'lat', 
                                                                                    'speed', 'prototype', 'category'])
raw_pqt = pd_reset_index(raw_pqt)
# deperacated
# raw = dd.read_hdf('dataset/raw_et_shenzhen_trajectory/*.hdf', 'trajectory')


# In[9]:


from dask.diagnostics import ProgressBar
with ProgressBar():
    raw_pqt.persist(scheduler='processes')


# ### category value counts

# In[ ]:


with ProgressBar():
    category_counts = raw_pqt.groupby('plate')['category'].first().value_counts().compute(scheduler='processes')
category_counts


# ### prototype value counts

# In[ ]:


with ProgressBar():
    prototype_counts = raw_pqt.groupby('plate')['prototype'].first().value_counts().compute(scheduler='processes')
prototype_counts


# ### company value counts

# In[ ]:


raw_company = dd.read_parquet('dataset/raw_trajectory_et_shenzhen_20180724to30.pqt', columns=['plate', 'company'])
with ProgressBar():
    company_counts = raw_company.groupby('plate')['company'].first().value_counts().compute(scheduler='processes')
company_counts


# ### color value counts

# In[ ]:


with ProgressBar():
    color_counts = raw_pqt.groupby('plate')['color'].first().value_counts().compute(scheduler='processes')
color_counts


# ### passenger value counts

# In[ ]:


raw_pqt['passenger'].value_counts().compute()


# ## filter et

# In[10]:


prototype_mask = raw_pqt['prototype'].isin(['比亚迪e6', '比亚迪E6'])
filtered_et = raw_pqt.loc[(raw_pqt['category']=='营运') & prototype_mask]


# ### et color value counts

# In[ ]:


with ProgressBar():
    filtered_color_counts = filtered_et['color'].value_counts().compute()
filtered_color_counts


# ### et speed value counts

# In[ ]:


with ProgressBar():
    speed_counts = filtered_et['speed'].value_counts().compute(scheduler='processes')
speed_counts


# ## drop useless columns

# In[11]:


filtered_et = filtered_et.drop(columns=['prototype', 'category'])


# ## transfer time format

# In[12]:


# 2018-07-24T00:00:24.000Z
filtered_et['timestamp'] = dd.to_datetime(filtered_et['timestamp'], errors='coerce', format='%Y-%m-%d %H:%M:%S').dt.tz_localize(None)


# ## sort, dask doesn't support sort_values, using pandas

# In[13]:


with ProgressBar():
    washed = filtered_et.compute(scheduler='processes')


# In[10]:


washed.sort_values(['plate', 'timestamp'], inplace=True)


# In[25]:


daskwashed = dd.from_pandas(washed, chunksize=524288)


# In[35]:


compressed_daskwashed = daskwashed.astype({'speed': np.uint8})


# ## drop rows cotaining invalid values

# In[62]:


zero_coordinates_mask = ((compressed_daskwashed['lng'] < 113.764635)
                         | (compressed_daskwashed['lng'] > 114.608972)
                         | (compressed_daskwashed['lat'] < 22.454727)
                         | (compressed_daskwashed['lat'] > 22.842654))
trajectory_in_bbox = compressed_daskwashed.loc[~zero_coordinates_mask]
with ProgressBar():
    trajectory_in_bbox.to_parquet('dataset/trajectory_et_sz_201807to30.pqt')


# # Charging event extraction

# In[24]:


temp_trajectory = dd.read_parquet('dataset/trajectory_et_sz_201807to30.pqt')


# In[27]:


temp_trajectory.head(10)


# ## info complementation

# In[4]:


import numpy as np
import dask.dataframe as dd
from dask.diagnostics import ProgressBar


# ### add dis2pre, dur2pre

# In[15]:


from utils.trajectory import haversine_dask
raw_trajectory = dd.read_parquet('dataset/trajectory_et_sz_201807to30.pqt')
raw_trajectory['dis2pre'] = haversine_dask(raw_trajectory['lng'].shift(), raw_trajectory['lat'].shift(),
                                           raw_trajectory['lng'], raw_trajectory['lat']).astype(np.float32)
raw_trajectory['dur2pre'] = (raw_trajectory['timestamp']
                             - raw_trajectory['timestamp'].shift()).dt.total_seconds().astype(np.float32)
first_vehicle_mask = raw_trajectory['plate'] != raw_trajectory['plate'].shift()
raw_trajectory['dis2pre'] = raw_trajectory['dis2pre'].mask(first_vehicle_mask, np.nan)
raw_trajectory['dis2pre'] = raw_trajectory['dur2pre'].mask(first_vehicle_mask, np.nan)


# ### add big_dur, valid, stop

# In[16]:


# 4. Add big_dur flag
raw_trajectory['big_dur'] = raw_trajectory['dur2pre'] > 1800
# 5. Add valid flag
raw_trajectory['valid'] = ~(raw_trajectory['big_dur'] & (raw_trajectory['dis2pre'] > 500))
# 6. Add stop flag
raw_trajectory['stop'] = (((raw_trajectory['speed'] == 0) & ~raw_trajectory['big_dur']) |
                          (raw_trajectory['big_dur'] & raw_trajectory['valid']))


# In[17]:


raw_trajectory


# In[21]:


with ProgressBar():
    ret = raw_trajectory.persist(scheduler='threads')


# In[23]:


with ProgressBar():
    ret.compute()

