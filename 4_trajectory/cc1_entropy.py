#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import csv
import networkx as nx
import matplotlib.pyplot as plt
import calendar
from collections import Counter
import math

geopath = 'group1_209736.csv'
geo = pd.read_csv(geopath , encoding = 'utf8' , usecols = ['user_id','tweet_bounding_box','tweet_created_at','goodtime','tweet_geo_state','tweet_geo_city'])

#################################'tweet_geo_location'###################
geo['tweet_geo_location'] = geo['tweet_geo_city'] + '+' + geo['tweet_geo_state']
geo = geo.drop(['tweet_geo_city','tweet_geo_state'],axis = 1)
#################################'tweet_geo_location'###################
#################################create bounding_box_df#################
bounding_box_df = geo.drop_duplicates('tweet_geo_location')
bounding_box_df = bounding_box_df.drop(['user_id','goodtime','tweet_created_at'], axis = 1)
dict_bbox = bounding_box_df.set_index('tweet_geo_location').T.to_dict('list')
dictlist_location = []
dictlist_bbox = []
for location,bbox in dict_bbox.items():
    dictlist_location.append(location)
    dictlist_bbox.append(bbox)
#    input: somevalue
#    somevalue_index = dictlist_location.index(somevalue)
#    output: dictlist_bbox[somevalue_index]
#################################create bounding_box_df#################

def entropy(alist):
    sum_list = sum(alist)
    alist = [x/sum_list for x in alist]
    alist = [x**x for x in alist]
    alist = [-math.log(x) for x in alist]
    return sum(alist)

################################ entropy ###############

entropy_dict = dict()

location_list = geo['tweet_geo_location'].tolist()
location_cnts = Counter(location_list)

cnts_df = pd.DataFrame.from_dict(location_cnts, orient = 'index', columns = ['score'])
cnts_df_copy = cnts_df
cnts_df = cnts_df.reset_index().rename(columns = {'index':'location'})


for somelocation in cnts_df['location']:
    somelocation_df = geo[geo['tweet_geo_location'].isin([somelocation])]
    
    user_list = somelocation_df['user_id'].tolist()
    user_to_somelocation_cnts = Counter(user_list)
    
    user_cnts_df = pd.DataFrame.from_dict(user_to_somelocation_cnts, orient = 'index', columns = ['score'])
    user_cnts_df = user_cnts_df.reset_index().rename(columns = {'index':'user_id'})
    user_cnts_list = user_cnts_df['score'].tolist()
    
    value = entropy(user_cnts_list)
    adic = {somelocation:value}
    entropy_dict.update(adic)

######################################################print(entropy_dict)

final_df = pd.DataFrame.from_dict(entropy_dict, orient = 'index', columns = ['entropy'])
final_df = final_df.join(cnts_df_copy)
final_df = final_df.reset_index().rename(columns = {'index':'location'})
final_df = final_df.sort_values(by=['entropy'], ascending=False)

print('final_df.shape[0]')
print(final_df.shape[0])

final_df = final_df[~final_df['entropy'].isin([0])]

print('final_df.shape[0]--2')
print(final_df.shape[0])

final_df['counter'] = range(len(final_df))

final_df['score'] = final_df['score'].map(lambda x: math.log(x))

final_df.head(50)

#plt.rcParams['figure.figsize'] = (12, 8)
#plt.scatter(final_df["counter"], final_df["entropy"], s = 15)

xxx = final_df['counter']
y_entropy = final_df['entropy']
y_cnts = final_df['score']

fig = plt.figure()
ax = fig.add_subplot(111)
#plt.scatter(xxx, y_entropy, s = 15, label = 'entropy')
ax.plot(xxx, y_entropy, '-', label = 'entropy')
ax2 = ax.twinx()
#ax2.plot(xxx, y_cnts, '-r', label = 'cnts')
plt.scatter(xxx, y_cnts, s = 15, label = 'cnts')
ax.legend(loc=0)
ax.grid()
ax.set_xlabel("numbers")
ax.set_ylabel(r"entropy")
ax2.set_ylabel(r"cnts")
ax.set_ylim(0,5.5)
ax2.set_ylim(0,10)
ax2.legend(loc=0)
#plt.savefig('0.png')

plt.show


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




