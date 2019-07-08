# -*- coding: utf-8 -*-
"""
@author: Yin Wan
@environment: Python 3.7
@file: scrape_followers_friends.py
@time: 2019/5/8 
"""
import tweepy
import pandas as pd

def filter_geo(user_input_csv):
    '''
    To filter users tweeting with geo location.
    :param user_input_csv: CSV file of user's id with columns:
                            'user_id', 'id_count', 'tweets_searched', 'tweet_with_geo_location'
    :return: df_filter: The dataframe of users'id after filtering by the column of 'tweet_with_geo_location'.
    '''
    df = pd.read_csv(user_input_csv, encoding='utf-8')
    df['friends_id'] = None
    df_filter = df[df['tweet_with_geo_location'] == True]
    df_filter.reset_index(drop=True, inplace=True)
    return df_filter

def get_followers_id(user_id):
    '''
    Get at most 5000 followers id of the user.
    :param user_id:
    :return:
    '''
    try:
        return api.followers_ids(user_id=user_id, count=5000)
    except:
        return None

def get_friends_id(user_id):
    '''
    Get at most 5000 friends id of the user.
    :param user_id:
    :return:
    '''
    try:
        return api.friends_ids(user_id=user_id, count=5000)
    except:
        return None


if __name__ == "__main__":

    # Since the account is private, the keys and secrets will not be shown here.
    # If any questions, please feel free to contact me.

    consumer_key = "*****Your Consumer Key*****"
    consumer_secret = "*****Your Consumer Secret*****"
    access_token_key = "*****Your Access Token Key*****"
    access_token_secret = "*****Access Token Secret*****"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    api = tweepy.API(auth,proxy="127.0.0.1:1080", wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    alist = ['1.csv','2.csv','3.csv','4.csv','5.csv','6.csv']
    for j in range(len(alist)):
        df_geo = filter_geo(alist[j])
        # Method 1:
        # df_geo['followers_id'] = list(map(lambda user_id: get_followers_id(int(user_id)), df_geo['user_id']))
        # df_geo['friends_id'] = list(map(lambda user_id: get_friends_id(int(user_id)), df_geo['user_id']))

        # Method 2:
        count = 0
        for user_id in df_geo['user_id']:
            if count % 100 == 0:
                print("%d / %d user_ids have been completed." % (count, df_geo.iloc[:,0].size))
            friends_list = get_friends_id(int(user_id))
            if friends_list != None:
                df_geo.loc[count, 'friends_id'] = str(friends_list)
            count += 1

        user_output_csv = 'group1_user2_1_' + str(j+1)+ '_'+ str(count) + '_searched_friends.csv'
        df_geo.to_csv(user_output_csv, index=0, encoding='utf-8')
