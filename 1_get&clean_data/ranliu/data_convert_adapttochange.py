#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import csv
import networkx as nx
import matplotlib.pyplot as plt
import calendar

G = nx.Graph()

grouppath = '2_group1_fuoutcome.csv'
geopath = 'group1_209736.csv' # 改成想要的文件
group = pd.read_csv(grouppath , encoding = 'utf8')
geo = pd.read_csv(geopath , encoding = 'utf8', usecols=['user_id','tweet_created_at','tweet_bounding_box','tweet_geo_city','tweet_geo_state'])


#################################'tweet_created_at'#####################
def timecov(df):
    def prmon(x):  #月份转换
        if list(calendar.month_abbr).index(x) >= 10: return str(list(calendar.month_abbr).index(x))
        else: return '0' + str(list(calendar.month_abbr).index(x))
    df['tweet_created_at'] = df['tweet_created_at'].str.split(' ')
    df['tweet_created_at_month'] = df['tweet_created_at'].str[1]
    df['tweet_created_at_month'] = df['tweet_created_at_month'].apply(prmon) #apply函数好像不能识别df的一部分，只能新建一列再识别
    df['goodday'] = df['tweet_created_at'].str[5]+df['tweet_created_at_month']+df['tweet_created_at'].str[2]
    df['goodtime'] = df['tweet_created_at'].str[3]
    df['goodtime'] = df['goodtime'].str.replace(':','')
    df['goodtime'] = df['goodday']+df['goodtime']
    return df
timecov(geo)
geo = geo.drop(['tweet_created_at_month','goodday'],axis = 1)
#################################'tweet_created_at'#####################
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
#################################group users only and count#############
group = group[group['user'].str[0].isin(['i'])]
group['user'] = group['user'].str[1:] # cnt 4508
group_cnt = pd.DataFrame(group['community'].value_counts())  # can be used later
group_cnt = group_cnt.rename(columns = {'community':'count'})
#print(group_cnt)
#group = group.groupby(['community']).get_group(16)  # 这里可以单独一个个表来，也可以写个循环把23个都导出，但是总是出bug所以先单个来
#group = group[group['user_id'].isin(geo['user_id'])]
#################################group users only and count#############



listexp = geo['user_id'].tolist() # 文件中所有user构成的list
geo = geo[geo['user_id'].isin(['104249727'])] # 单个用户时
#geo = geo[geo['user_id'].isin(listexp)]
print('geo cnts')
print(geo.shape[0])
user_clean = geo.drop_duplicates('user_id')
print('user cnts')
print(user_clean.shape[0])
useridlist = user_clean['user_id'].tolist()  # create 单纯的userid列表，无重复 to make sure geo and group have same userid


final_data = pd.DataFrame(columns=['start_point', 'end_point']) # 新建dataframe用于存储数据

###################################以下为路径读取循环##############
for someid in useridlist:  # 743633
    single_df = geo.groupby(['user_id']).get_group(someid)
    single_df = single_df.sort_values(by=['goodtime'])
    single_list = single_df['tweet_geo_location'].tolist()
    a = len(single_list)
    start_list = single_list[0:a-1]
    end_list = single_list[1:a]
    single_data = pd.DataFrame({'start_point': start_list,'end_point':end_list})
    final_data = final_data.append(single_data)
##################################final_data start-end##########
############################## add bbox ########################
final_data['start_point_bbox'] = final_data['start_point']
final_data['end_point_bbox'] = final_data['end_point']

for location in final_data['start_point']:
    location_index = dictlist_location.index(location)
    final_data.start_point_bbox[final_data.start_point == location] = dictlist_bbox[location_index]
for location in final_data['end_point']:
    location_index = dictlist_location.index(location)
    final_data.end_point_bbox[final_data.end_point == location] = dictlist_bbox[location_index]
############################## add bbox ########################   

final_data.to_csv('i104249727_finaldata.csv', encoding="utf_8_sig",index=False) ## 改文件名字！

print(final_data.shape[0])

final_data['path'] = final_data['start_point'] + '+' + final_data['end_point']
final_cnt_data = pd.DataFrame(final_data['path'].value_counts())
final_cnt_data = final_cnt_data.rename(columns = {'path':'count'})  ### path 的计数表，可供使用

final_cnt_data.to_csv('group1_community3_finalcntdata.csv', encoding="utf_8_sig",index=False)

#print(range(len(final_data)))


final_startlist = final_data['start_point'].tolist()
final_endlist = final_data['end_point'].tolist()

for i in range(len(final_startlist)):    
    ix = final_startlist[i]
    iy = final_endlist[i]
    G.add_edge(ix,iy)
    

plt.rcParams['figure.figsize'] = (100, 100)
nx.draw(G, node_size = 200, font_size = 40, width = 5, with_labels=True)
#plt.savefig("100_100_test.png")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




