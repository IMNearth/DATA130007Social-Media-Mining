#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import chardet
import pandas as pd
import calendar

data_tweets = pd.read_csv('group2_tweets_54657.csv',encoding = 'utf8') #数据名称为data


### 先整改时间函数
def timecov(df):
    def prmon(x):  #月份转换
        if list(calendar.month_abbr).index(x) >= 10: return str(list(calendar.month_abbr).index(x))
        else: return '0' + str(list(calendar.month_abbr).index(x))
    df['tweet_created_at_new'] = df['tweet_created_at'].str.split(' ')
    df['tweet_created_at_month'] = df['tweet_created_at_new'].str[1]
    df['tweet_created_at_month'] = df['tweet_created_at_month'].apply(prmon) #apply函数好像不能识别df的一部分，只能新建一列再识别
    df['goodtime'] = df['tweet_created_at_new'].str[5]+df['tweet_created_at_month']+df['tweet_created_at_new'].str[2]
    return df

timecov(data_tweets)
data_tweets = data_tweets.drop(['tweet_created_at_new','tweet_created_at_month'],axis = 1)

###整改完毕

###再整改经纬度数据

def find_centroid(row):
    '''
    To find the centroid of a polygonal bounding box of longitude, latitude coordinates.
    :param row: A bounding box.
    :return: centroid_list: A list: [longitude, latitude] coordinates
    '''
    try:
        row_ = eval(row)
        lst_of_coords = [item for sublist in row_ for item in sublist]
        longitude = [p[0] for p in lst_of_coords]
        latitude = [p[1] for p in lst_of_coords]
        centroid_list = [sum(longitude) / float(len(longitude)), sum(latitude) / float(len(latitude))]
        return centroid_list
    except:
        return None
    
data_tweets['tweet_bounding_box'] = data_tweets['tweet_bounding_box'].apply(find_centroid)

###整改完毕！

###将他们按国家好不好，state好不好来分个类吧！
'''
def country(x):
    if x != 'US' : return 'None'
    else: return 'US'
data_tweets['country_code'] = data_tweets['country_code'].apply(country)
'''

data_tweets.loc[data_tweets.country_code != 'US', 'tweet_geo_location'] = 'None, None' #其他国家的数据不读取

data_tweets['tweet_geo_location_new'] = data_tweets['tweet_geo_location'].str.split(', ')
data_tweets['tweet_geo_city'] = data_tweets['tweet_geo_location_new'].str[0]
data_tweets['tweet_geo_state'] = data_tweets['tweet_geo_location_new'].str[1] #将geo_location分成city和state
data_tweets = data_tweets.drop(['tweet_geo_location','tweet_geo_location_new'], axis = 1)

#原数据分为两个df，分别名为countrywrong和countryright，其中countrywrong将返回country None state None city None

countrywrong = data_tweets[~data_tweets['country_code'].isin(["US"])] #countrywrong已完成，先不理你啦！
countryright = data_tweets[data_tweets['country_code'].isin(["US"])]

#以下处理countryright

def goodstate(x):
    goodstate = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "HI", "ID",                                     "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO",                                     "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA",                                     "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]
    if (x in goodstate): return x
    else: return 'fix'
countryright['tweet_geo_state'] = countryright['tweet_geo_state'].apply(goodstate)

#countryright分为两个df，分别名为goodstate和badstate

goodstate = countryright[~countryright['tweet_geo_state'].isin(["fix"])] #goodstate也已经完成啦！
badstate = countryright[countryright['tweet_geo_state'].isin(["fix"])]


badstate.to_csv('badstate_group2_lr.csv', encoding="utf_8_sig",index=False) #防止乱码的方式
goodstate.to_csv('goodstate_group2_lr.csv', encoding="utf_8_sig",index=False) #防止乱码的方式
countrywrong.to_csv('countrywrong_group2_lr.csv', encoding="utf_8_sig",index=False) #防止乱码的方式

'''

所有文件被分为三类
：countrywrong 国家 None，city None，state None
：goodstate 国家 US，city 对，state 对

   #以上两个文件没有导出，请自行导出
   
：badstate 请导出为csv并且使用part2程序处理

'''



#countrywrong.head()


# In[6]:


import csv
import chardet
import pandas as pd
import calendar

data1 = pd.read_csv('goodstate_lr.csv',encoding = 'utf8') #数据名称为data
data2 = pd.read_csv('badstate_lr.csv',encoding = 'utf8') #数据名称为data
data3 = pd.read_csv('countrywrong_lr.csv',encoding = 'utf8') #数据名称为data

print(data1.shape[0])
print(data2.shape[0])
print(data3.shape[0])

print(186332+33611)


# In[ ]:


235259

