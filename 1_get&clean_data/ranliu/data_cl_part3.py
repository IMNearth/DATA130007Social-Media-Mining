#!/usr/bin/env python
# coding: utf-8

# In[81]:


import pandas as pd
import os

#df_empty = pd.Dataframe()

df1 = pd.read_csv('countrywrong_group2_lr.csv',encoding = 'utf8') #数据名称为data
df2 = pd.read_csv('goodstate_group2_lr.csv',encoding = 'utf8') #数据名称为data
df3 = pd.read_csv('badstate_group2_lr_cl.csv',encoding = 'utf8') #数据名称为data

'''
df3['tweet_bounding_box'] = df3['tweet_bounding_box'].str.replace('[','')
df3['tweet_bounding_box'] = df3['tweet_bounding_box'].str.replace(']','')
df3['tweet_bounding_box'] = df3['tweet_bounding_box'].str.split(')
'''


#for item in df3['tweet_bounding_box']:
#    for i, v in enumerate(item): item[i] = float(v)

for item in df3['tweet_bounding_box']:
    item = map(float, item)

#df1 = df1.append(df2)
#df1 = df1.append(df3)


df3.head()

#df1.to_csv('group2_sep.csv', encoding="utf_8_sig",index=False) #防止乱码的方式


# In[ ]:





# In[ ]:




