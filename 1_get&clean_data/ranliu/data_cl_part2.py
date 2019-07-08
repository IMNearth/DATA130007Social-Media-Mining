#!/usr/bin/env python
# coding: utf-8

# In[1]:


from googleplaces import GooglePlaces
import googlemaps
import json

import sys
import importlib
importlib.reload(sys)


def statereg(x):
    if x == 'Alabama'.upper(): return 'AL'
    elif x == 'Alaska'.upper(): return 'AK'
    elif x == 'Arizona'.upper(): return 'AZ'
    elif x == 'Arkansas'.upper(): return 'AR'
    elif x == 'California'.upper(): return 'CA'
    elif x == 'Colorado'.upper(): return 'CO'
    elif x == 'Connecticut'.upper(): return 'CT'
    elif x == 'Delaware'.upper(): return 'DE'
    elif x == 'Florida'.upper(): return 'FL'
    elif x == 'Georgia'.upper(): return 'GA'
    elif x == 'Hawaii'.upper(): return 'HI'
    elif x == 'Idaho'.upper(): return 'ID'
    elif x == 'Illinois'.upper(): return 'IL'
    elif x == 'Indiana'.upper(): return 'IN'
    elif x == 'Iowa'.upper(): return 'IA'
    elif x == 'Kansas'.upper(): return 'KS'
    elif x == 'Kentucky'.upper(): return 'KY'
    elif x == 'Louisiana'.upper(): return 'LA'
    elif x == 'Maine'.upper(): return 'ME'
    elif x == 'Maryland'.upper(): return 'MD'
    elif x == 'Massachusetts'.upper(): return 'MA'
    elif x == 'Michigan'.upper(): return 'MI'
    elif x == 'Minnesota'.upper(): return 'MN'
    elif x == 'Mississippi'.upper(): return 'MS'
    elif x == 'Missouri'.upper(): return 'MO'
    elif x == 'Montana'.upper(): return 'MT'
    elif x == 'Nebraska'.upper(): return 'NE'
    elif x == 'Nevada'.upper(): return 'NV'
    elif x == 'New hampshire'.upper(): return 'NH'
    elif x == 'New jersey'.upper(): return 'NJ'
    elif x == 'New mexico'.upper(): return 'NM'
    elif x == 'New York'.upper(): return 'NY'
    elif x == 'North Carolina'.upper(): return 'NC'
    elif x == 'North Dakota'.upper(): return 'ND'
    elif x == 'Ohio'.upper(): return 'OH'
    elif x == 'Oklahoma'.upper(): return 'OK'
    elif x == 'Oregon'.upper(): return 'OR'
    elif x == 'Pennsylvania'.upper(): return 'PA'
    elif x == 'Rhode island'.upper(): return 'RI'
    elif x == 'South carolina'.upper(): return 'SC'
    elif x == 'South dakota'.upper(): return 'SD'
    elif x == 'Tennessee'.upper(): return 'TN'
    elif x == 'Texas'.upper(): return 'TX'
    elif x == 'Utah'.upper(): return 'UT'
    elif x == 'Vermont'.upper(): return 'VT'
    elif x == 'Virginia'.upper(): return 'VA'
    elif x == 'Washington'.upper(): return 'WA'
    elif x == 'West Virginia'.upper(): return 'WV'
    elif x == 'Wisconsin'.upper(): return 'WI'
    elif x == 'Wyoming'.upper(): return 'WY'
    else: return 'None'


class GoogleMaps(object):
    """提供google maps服务"""

    def __init__(self):

        self._GOOGLE_MAPS_KEY = "AIzaSyCOevFB5iJXzZ0-lEBAlpHpMKVQucXGOes"
        self._Google_Places = GooglePlaces(self._GOOGLE_MAPS_KEY)
        self._Google_Geocod = googlemaps.Client(key=self._GOOGLE_MAPS_KEY)

    def _reverse_geocode(self, lat, lng, language=None):
        """
        根据经纬度请求google API获取坐标信息,返回信息:param lat: 纬度:param lng:经度:param language:语言:return:
        """
        # 根据经纬度获取地址信息 pincode
        list_reverse_geocode_result = self._Google_Geocod.reverse_geocode((lat, lng), language=language)
        # print json.dumps(list_reverse_geocode_result, indent=4)
        return list_reverse_geocode_result

    def _return_reverse_geocode_info(self, lat, lng, language=None):
        """
        整理信息:param lat:纬度:param lng:经度:param language:语言:return:
        """
        list_reverse_geocode = self._reverse_geocode(lat, lng, language=language)
        if list_reverse_geocode:
            city = ''
            pincode = ''
            route = ''
            neighborhood = ''
            sublocality = ''
            administrative_area_level_1 = ''
            country = ''
            street_number = ''
            # 全名地址
            formatted_address = list_reverse_geocode[0]['formatted_address']
            for address_info in list_reverse_geocode[0]['address_components']:
                # 城市标识为locality
                if 'locality' in address_info['types']:
                    city = address_info['long_name']
                # 邮政编码标识为postal_code
                elif 'postal_code' in address_info['types']:
                    pincode = address_info['long_name']
                # 街道路
                elif 'route' in address_info['types']:
                    route = address_info['long_name']
                # 相似地址名
                elif 'neighborhood' in address_info['types']:
                    neighborhood = address_info['long_name']
                # 地区名
                elif 'sublocality' in address_info['types']:
                    sublocality = address_info['long_name']
                # 省份
                elif 'administrative_area_level_1' in address_info['types']:
                    administrative_area_level_1 = address_info['long_name']
                # 国家
                elif 'country' in address_info['types']:
                    country = address_info['long_name']
                # 门牌号
                elif 'street_number' in address_info['types']:
                    street_number = address_info['long_name']
            return {'city': city, 'pincode': pincode, 'route': route, 'neighborhood': neighborhood,
                    'sublocality': sublocality, 'administrative_area_level_1': administrative_area_level_1,
                    'country': country, 'formatted_address': formatted_address, 'street_number': street_number}
        else:
            return None

    def get_pincode_city(self, latlng, language=None):
        """
        根据经纬度获取该地区详细信息:param lat: 纬度:param lng: 经度:return:
        """
        lat = latlng[1]
        lng = latlng[0]
        reverse_geocode_info = self._return_reverse_geocode_info(lat, lng, language=language)
        if reverse_geocode_info:
            return [reverse_geocode_info['country'], reverse_geocode_info['administrative_area_level_1'], reverse_geocode_info['city']]
        else:
            return None
        
google_maps = GoogleMaps()


####要开始操作了！

import csv
import chardet
import pandas as pd
import calendar


# 请更改为part1程序中导出的csv文件
badstate = pd.read_csv('badstate_group2_lr.csv',encoding = 'utf8')

def covertbox(x):
    boxnew = []
    x = x.strip('[')
    x = x.strip(']')
    boxnew = x.split(', ')
    return boxnew
    
badstate['tweet_bounding_box'] = badstate['tweet_bounding_box'].apply(covertbox)
badstate['find_state'] = badstate['tweet_bounding_box'].apply(google_maps.get_pincode_city) # example[-105.3017759, 39.953552]

badstate['find_state_state'] = badstate['find_state'].str[1]
badstate['find_state_state'] = badstate['find_state_state'].str.upper()

badstate['tweet_geo_state'] = badstate['find_state_state'].apply(statereg)
badstate['tweet_geo_city'] = badstate['find_state'].str[2]

badstate.loc[badstate.tweet_geo_state == 'None', 'tweet_geo_city'] = 'None'
badstate = badstate.drop(['find_state_state', 'find_state'], axis = 1)

# 此时的badstate已处理好！导出即可使用！

badstate.to_csv('badstate_group2_lr_cl.csv', encoding="utf_8_sig",index=False) #防止乱码的方式


# In[ ]:




