# -*- coding: utf-8 -*-
"""
@author: Yin Wan
@environment: Python 3.7
@file: filter_social.py
@time: 2019/5/20 
"""
import re
import pandas as pd


if __name__ == "__main__":
    df_group1 = pd.read_csv('group1_user_23753_searched.csv', usecols=['user_id'], encoding='utf-8')
    df_group2 = pd.read_csv('group2_user_10452_unsearched.csv', usecols=['user_id'], encoding='utf-8')
    user = pd.concat([df_group1, df_group2], sort=False, ignore_index=True)
    user.drop_duplicates(subset=['user_id'], keep='first', inplace=True)
    user.reset_index(drop=True, inplace=True)

    user = list(map(str, user['user_id']))

    df_social = pd.read_csv('group1_user_4918_searched_friends.csv',
                            usecols=['user_id', 'friends_id'], encoding='utf-8')
    for i in range(df_social.iloc[:,0].size):
        friends = df_social.loc[i, 'friends_id']
        d_friends = []
        if type(friends) == str:
            t = re.findall(r"[0-9]+", friends)
            for item in t:
                if item in user:
                    d_friends.append(item)
        df_social.loc[i, 'friends_id'] = str(d_friends)
    df_social.to_csv('group1_user_4918_searched_friends_filter.csv', index=0, encoding='utf-8')

