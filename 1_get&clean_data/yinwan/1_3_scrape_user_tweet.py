# -*- coding: utf-8 -*-
"""
@author: Yin Wan
@environment: Python 3.7
@file: scrape_user_tweet.py
@time: 2019/5/8 
"""
import tweepy
import pandas as pd
from pymongo import MongoClient

def get_user_tweet(db,df):
    '''
    Get at most 200 tweets of the Twitter user.
    The JSON data of the tweets will be stored in collection called 'training_user_tweets_collection' of
    the database called 'training_tweets' in MongoDB.
    To export the JSON file, please run the following command:
        mongoexport --db training_tweets --collection training_user_tweets_collection --out training_user_tweets.json
    :param db: The database in MongoDB where to store the tweet data.
    :param df: The dataframe of users'id  with columns: 'user_id', 'tweets_searched'.
                If the tweets of the user is protected, the value of 'tweets_searched' will turn to 'Protected'.
    :return:df: The dataframe of users'id after searching their tweets.
    '''
    df['tweet_with_geo_location'] = False
    for i in range(df.iloc[:,0].size):
        if i % 100 == 0:
            print("%d / %d user_ids have been completed." % (i, df.iloc[:, 0].size))
        user_id = int(df.loc[i,'user_id'])
        tweets_searched = df.loc[i,'tweets_searched']
        if tweets_searched == False:
            try:
                tweets = api.user_timeline(id=user_id, count=200)
            except:
                df.loc[i, 'tweets_searched'] = 'PROTECTED'
                continue
            for tweet in tweets:
                if tweet._json and tweet._json['place'] != None:
                    db.training_user_tweets_collection2.insert_one(tweet._json)
                    if df.loc[i,'tweet_with_geo_location'] == False:
                        df.loc[i,'tweet_with_geo_location'] = True
            df.loc[i, 'tweets_searched'] = True

    return df


if __name__ == "__main__":

    # Since the account is private, the keys and secrets will not be shown here.
    # If any questions, please feel free to contact me.

    consumer_key = "*****Your Consumer Key*****"
    consumer_secret = "*****Your Consumer Secret*****"
    access_token_key = "*****Your Access Token Key*****"
    access_token_secret = "*****Access Token Secret*****"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)
    api = tweepy.API(auth)

    client = MongoClient()
    db = client.training_tweets

    df = pd.read_csv('group2_user_10452_.csv', encoding='utf-8')

    get_user_tweet(db,df)
    df.to_csv('group2_user_10452_searched.csv', index=0, encoding='utf-8')

