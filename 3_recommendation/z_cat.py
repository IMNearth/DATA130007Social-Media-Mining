# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-22 18:18
@environment : python 3.6
@author      : zhangjiwen
@file        : cat.py
"""
import pandas as pd
from collections import Counter
import numpy as np
import random


def build_formal_samples(geo_dt, usr_count_dict, emb_dt, userAndgeo, city_list):
    c_train, c_test, c_vali = 0, 0, 0

    path1 = '/Users/zhangjiwen/Desktop/twitter/5_recommendation_on_linux/structure_2/data/formal_train.txt'
    path2 = '/Users/zhangjiwen/Desktop/twitter/5_recommendation_on_linux/structure_2/data/formal_test.txt'
    path3 = '/Users/zhangjiwen/Desktop/twitter/5_recommendation_on_linux/structure_2/data/formal_validation.txt'
    with open(path1, 'w') as train_file, open(path2, 'w') as test_file, open(path3, 'w') as validation_file:
        for usr, c in usr_count_dict.items():
            if c_train%1000 == 0:
                print("\t train:%d, test:%d, validation:%d" % (c_train, c_test, c_vali))

            wk_dt = geo_dt[geo_dt['user_id'] == usr]
            index = list(wk_dt.index)
            random.shuffle(index)

            cities = list(wk_dt['tweet_geo_city'])
            cities = [''.join(city.split(' ')) for city in cities]
            if c >= 10:
                c1 = int(c * 0.6)
                c2 = int(c * 0.2)
                c3 = c - c2 - c1

                sampling(0, c1, index, train_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=1,)
                c_train += c1
                sampling(c1, c1 + c2, index, test_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=2,)
                c_test += c2
                sampling(c1 + c2, c, index, validation_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=2,)
                c_vali += c3
            else:
                sampling(0, c, index, train_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=1, )
                c_train += c

        print("Has Write \n\t train file : %d\n\t test file : %d\n\t validation file : %d" % (c_train, c_test, c_vali))


def build_test_samples(geo_dt, usr_count_dict, emb_dt, userAndgeo, city_list):
    c_train, c_test, c_vali = 0, 0, 0

    path1 = '/Users/zhangjiwen/Desktop/twitter/4_Recommendation_2/结构化2/data/cmp_test_train.txt'
    path2 = '/Users/zhangjiwen/Desktop/twitter/4_Recommendation_2/结构化2/data/cmp_test_test.txt'
    path3 = '/Users/zhangjiwen/Desktop/twitter/4_Recommendation_2/结构化2/data/cmp_test_validation.txt'
    with open(path1, 'w') as train_file, open(path2, 'w') as test_file, open(path3, 'w') as validation_file:
        for usr, c in usr_count_dict.items():
            if c_train % 1000 == 0:
                print("\t train:%d, test:%d, validation:%d" % (c_train, c_test, c_vali))
            if c >= 10:
                c1 = int(c * 0.6)
                c2 = int(c * 0.2)
                # c3 = c - c2 - c1
                wk_dt = geo_dt[geo_dt['user_id'] == usr]
                index = list(wk_dt.index)
                random.shuffle(index)

                cities = list(wk_dt['tweet_geo_city'])
                cities = [''.join(city.split(' ')) for city in cities]

                sampling(0, c1, index, train_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=1, early_break=True)
                c_train += 2
                sampling(c1, c1+c2, index, test_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=1, early_break=True)
                c_test += 2
                sampling(c1 + c2, c, index, validation_file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr,
                         neg_num=1, early_break=True)
                c_vali += 2
        print("Has Write \n\t train file : %d\n\t test file : %d\n\t validation file : %d" % (c_train, c_test, c_vali))


def sampling(start, end, index, file, userAndgeo, emb_dt, geo_dt, city_list, cities, usr, neg_num = 1,
             early_break = False, feature_type = 1):
    for i in range(start, end):
        ind = index[i]
        label = '1 '
        city = geo_dt['tweet_geo_city'][ind].split(' ')
        city = ''.join(city)
        emb_city = emb_dt[userAndgeo.index(city), :]
        emb_user = emb_dt[userAndgeo.index(usr), :]
        emotion = geo_dt['emotion(0-1)'][ind]  # str
        if feature_type:
            feature = label
            j = 0
            for v in emb_user:
                feature += str(j) + ':' + v + ' '
                j += 1
            for v in emb_city:
                feature += str(j) + ':' + v + ' '
                j += 1
            '''
            feature += str(j) + ':' + emotion + ' '
            '''
            feature += user_map2_num[usr] + ':' + '1' + ' '
            feature += city_map2num[city] + ':' + '1' + '\n'
        else:
            feature = label + ' '.join(emb_user) + ' ' + ' '.join(emb_city) + ' ' + emotion
        file.write(feature)

        negative_cities = [city for city in city_list if city not in cities]
        random.shuffle(negative_cities)
        for k in range(0, neg_num):
            ng_city = negative_cities.pop()
            emb_city = emb_dt[userAndgeo.index(ng_city), :]
            emb_user = emb_dt[userAndgeo.index(usr), :]
            emotion = '0.5'
            if feature_type:
                feature = '0 '
                j = 0
                for v in emb_user:
                    feature += str(j) + ':' + v + ' '
                    j += 1
                for v in emb_city:
                    feature += str(j) + ':' + v + ' '
                    j += 1
                '''
                feature += str(j) + ':' + emotion + ' '
                '''
                feature += user_map2_num[usr] + ':' + '1' + ' '
                feature += city_map2num[ng_city] + ':' + '1' + '\n'
            else:
                feature = '0 ' + ' '.join(emb_user) + ' ' + ' '.join(emb_city) + ' ' + emotion
            file.write(feature)

        if early_break: break


if __name__ == "__main__":
    '''
    生成recommendation所需要的正样本和负样本
    
    用户数量：4508
    US城市数量：3878
    用户 -- 城市 unique 交互数量：20005 << 4509*3878 = 17 000 000+
            not unique 交互数量 209736 << 4509*3878 = 17 000 000+
    
    特征维度 ： 259 = 128*2 + 3 = 128(user) + 128(city) + 1(emotion of tweet -- microsoft) + 1(user_id) + 1(city_id)
               -- 标签 1(去过) vs 0(没去过)
               每行写入数据 260 = 259 + 1
    
    训练集 : 测试集 : 验证集合 == 6 : 2 : 2
    因此按照比例，从209736条正样本中为三个集合随机产生正样本
        此外，按照1：2的比例，从用户未去过的地点产生负样本 除了 emotion = 0.5， label = 0外，其他与正样本相同
    
    train set(6)
        最终数量：126489 * 3 = 379 467
    
    test set(2)
        最终数量：39760 * 3 = 119 280
    
    validation(2)
        最终数量：43487 * 3 = 130 461
    
    样本总数：629 208
    
    '''

    geo_dt = pd.read_csv('/Users/zhangjiwen/Desktop/twitter/data/modified/0_group1_US_sentiment.csv',
                      encoding='utf-8', dtype=str)

    usr_count_dict = Counter(list(geo_dt['user_id']))

    global user_map2_num
    global num_map2_usr

    user_map2_num = {}
    num_map2_usr = {}

    i = 257
    for sb in usr_count_dict.keys():
        user_map2_num[sb] = str(i)
        num_map2_usr[str(i)] = sb
        i += 1

    city_count_dict = Counter(list(geo_dt['tweet_geo_city']))

    global city_map2num

    city_map2num = {}

    for ct in city_count_dict.keys():
        ct = ct.strip().split(' ')
        ct = ''.join(ct)
        city_map2num[ct] = str(i)
        i += 1

    path = '/Users/zhangjiwen/Desktop/twitter/data/modified/2_group1_corpus_ivav_learned_0520.txt'
    emb_dt = np.loadtxt(path, dtype=str, skiprows=2)
    userAndgeo = list(emb_dt[:, 0])
    emb_dt = np.delete(emb_dt, 0, axis=1)
    #emb_dt = emb_dt.astype(float)

    city_list = list(city_map2num.keys())
    userAndgeo = [ item[1:] for item in userAndgeo]
    #build_test_samples(geo_dt, usr_count_dict, emb_dt, userAndgeo, city_list)
    build_formal_samples(geo_dt, usr_count_dict, emb_dt, userAndgeo, city_list)
