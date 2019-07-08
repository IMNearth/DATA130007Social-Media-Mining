#!/usr/bin/env python
# coding: utf-8

# In[11]:


import csv
import pandas as pd

data = pd.read_csv('part1_lr_full.csv', encoding ='utf8')

print(data.shape[0])

data = data[data['country_code'].isin(['US'])]
data = data[~data['tweet_geo_state'].isin(['None'])] #219940
data = data[~data['tweet_geo_city'].isin([''])] #210917
data.dropna(axis=0, how='any', inplace=True) #209736

print(data.shape[0])


data.to_csv('group1_209736.csv', encoding="utf_8_sig",index=False) #防止乱码的方式


# In[ ]:




