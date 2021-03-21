#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import yaml
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler, normalize
from utils import display, od_utils
from s1_preprocessing.hotspot.hotpots_discovery_utils import generate_cube_index
from s2_mobility.transit_prediction.s2_utility_XGBoost_train import kl


# In[2]:


# configure the working directory to the project root path
with open("../../config.yaml", "r", encoding="utf8") as f:
    conf = yaml.load(f, Loader=yaml.FullLoader)
os.chdir(conf["project_path"])
display.configure_pandas()

od = pd.read_parquet(conf['od']['raw1706_pqt'])
od = od_utils.filter_in_bbox(od)
od = generate_cube_index(od, m=100, n=200)
od_pairs_demand = pd.read_csv(conf['mobility']['transition']['utility_xgboost']['p2d']['result'])

percent_list = []
kl_list = []
for percentage_int in range(1, 100, 1):
    percentage = percentage_int / 100
    od_sample = od.sample(frac=percentage)
    demand = od_sample.groupby(['original_cube', 'destination_cube']).size().rename('sample')
    all_and_sample = pd.merge(od_pairs_demand, demand,
                              how='left',
                              left_on=['original_cube', 'destination_cube'], right_index=True).fillna(0.0001)
    kl_sample = all_and_sample.groupby("original_cube").apply(
        lambda x: kl(normalize(x["demand_17_et"].to_numpy().reshape(1, -1), norm='l1'),
                     normalize(x["sample"].to_numpy().reshape(1, -1), norm='l1'))
    )
    print(percentage, kl_sample.mean())
    percent_list.append(percentage)
    kl_list.append(kl_sample.mean())


# In[ ]:


font = {'family': 'Palatino Linotype',
        'weight': 'normal',
        'size': 15,
        }

import matplotlib.pyplot as plt
from brokenaxes import brokenaxes
import numpy as np
fig = plt.figure(figsize=(5.5, 4), dpi=600)
ax=brokenaxes(xlims=((-0.01,1.01),), ylims=((-0.05,3),(20.5, 21)), hspace=0.15, left = 0.175, bottom = 0.21)#hspace指两个断点之间的距离
#https://github.com/bendichter/brokenaxes/issues/20
ax.plot(percent_list, kl_list, label='Sampled data of taxis at 2017')
ax.plot(664/10000, 20.83, '.', label='Data of only EV taxis at 2014')
ax.plot(0.99, 0.21, '.', label='Data of taxis at 2014')
ax.legend(prop=font)
ax.set_xlabel('# of taxis / # of taxis at 2016', labelpad = 0, size=16, family='Palatino Linotype')
ax.set_ylabel('KL-Divergence', labelpad = 0, size=16, family='Palatino Linotype')
ax.tick_params(labelsize=12, )
# fig.tight_layout()
fig.show()
fig.savefig('kldiv_comparison.pdf')

