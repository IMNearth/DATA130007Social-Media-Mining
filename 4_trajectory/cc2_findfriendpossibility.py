#!/usr/bin/env python
# coding: utf-8

# sum program

# In[70]:


import pandas as pd
import csv
import networkx as nx
import matplotlib.pyplot as plt
import calendar
from collections import Counter
import math

geopath = 'group1_209736.csv'
frdpath = 'group1_user_4918_searched_friends_filter.csv'
frd = pd.read_csv(frdpath , encoding = 'utf8')
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
frd['user_id'] = frd['user_id'].apply(str) # 变成字符串
frd_new_df = frd.set_index('user_id') # 变成index
#print(frd_new_df.loc['12','friends_id']) # 用法
############################################### frds ###############

def find_frd(blist): # 在blist中找到朋友的个数(Note：没有除以2)
    their_frds = []
    for someone in blist: # someone：blist中的某人
        if someone in frd['user_id'].tolist():
            someone_frd = eval(frd_new_df.loc[someone , 'friends_id'])
            their_frds = their_frds + someone_frd
    frd_cnts = Counter(their_frds)
    frd_cnts_df = pd.DataFrame.from_dict(frd_cnts, orient = 'index', columns = ['cnts'])
    frd_cnts_df = frd_cnts_df.reset_index().rename(columns = {'index':'user'})
    frd_cnts_df = frd_cnts_df[frd_cnts_df['user'].isin(blist)]   
    frd_cnts_list = frd_cnts_df['cnts'].tolist()
    return sum(frd_cnts_list)

def find_place_frd(place): # 输入place（str）：返回概率值
    place_geo = geo[geo['tweet_geo_location'].isin([place])]
    place_geo = place_geo.drop_duplicates('user_id')
    place_geo['user_id'] = place_geo['user_id'].apply(str)
    user_place_list = place_geo['user_id'].tolist()
    c_value = (len(user_place_list) *(len(user_place_list) - 1))+0.0001
    a_value = find_frd(user_place_list)
    return a_value/c_value

def entropy(alist): # alist算出的熵
    sum_list = sum(alist)
    alist = [x/sum_list for x in alist]
    alist = [x**x for x in alist]
    alist = [-math.log(x) for x in alist]
    return sum(alist)

################################ entropy ###############

entropy_dict = dict() # 新建entropy字典
place_dict = dict() # 新建place字典，用于放置最终概率
location_list = geo['tweet_geo_location'].tolist()
location_cnts = Counter(location_list) # location计数字典

cnts_df = pd.DataFrame.from_dict(location_cnts, orient = 'index', columns = ['score'])
cnts_df_copy = cnts_df
cnts_df = cnts_df.reset_index().rename(columns = {'index':'location'}) # location-计数（score）dataframe：cntsdf

###################################################################

for somelocation in cnts_df['location']: # 不重复的location中的某个somelocation
    somelocation_df = geo[geo['tweet_geo_location'].isin([somelocation])]    
    user_list = somelocation_df['user_id'].tolist()
    user_to_somelocation_cnts = Counter(user_list) # somelocation的user计数字典
    user_cnts_df = pd.DataFrame.from_dict(user_to_somelocation_cnts, orient = 'index', columns = ['score'])
    user_cnts_df = user_cnts_df.reset_index().rename(columns = {'index':'user_id'}) #somelocation的user计数df（计数=score）
    user_cnts_list = user_cnts_df['score'].tolist()
    value = entropy(user_cnts_list) # somelocation中的userlist中的熵
    adic = {somelocation:value}
    entropy_dict.update(adic) # 字典更新（最终字典变成：所有location+对应+所有熵）
    place_value = find_place_frd(somelocation)
    bdic = {somelocation:place_value}
    place_dict.update(bdic)

######################################################print(entropy_dict)

final_df = pd.DataFrame.from_dict(entropy_dict, orient = 'index', columns = ['entropy'])
place_df = pd.DataFrame.from_dict(place_dict, orient = 'index', columns = ['place_frd_possi'])
final_df = final_df.join(cnts_df_copy)
final_df = final_df.join(place_df)
final_df = final_df.reset_index().rename(columns = {'index':'location'})
final_df = final_df.sort_values(by=['entropy'], ascending=False)


final_df.to_csv('final_df_sum.csv', encoding="utf_8_sig",index=False) #防止乱码的方式


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




