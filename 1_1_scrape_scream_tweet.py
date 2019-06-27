# -*- coding: utf-8 -*-
"""
@author: Yin Wan
@environment: Python 3.7
@file: scrape_scream_tweet.py
@time: 2019/4/15
@reference: https://github.com/shawn-terryah/Twitter_Geolocation
"""

import tweepy
import json
from pymongo import MongoClient


class StreamListener(tweepy.StreamListener):
    """tweepy.StreamListener is a class provided by tweepy used to access
    the Twitter Streaming API to collect tweets in real-time.
    """

    def __init__(self,api=None):
        super().__init__()
        self.counter = 0
        self.limit = 10

    def on_connect(self):
        """Called when the connection is made"""

        print("You're connected to the streaming server.")

    def on_error(self, status_code):
        """This is called when an error occurs"""

        print('Error: ' + repr(status_code))
        return False

    def on_data(self, data):
        """This will be called each time we receive stream data"""

        client = MongoClient()

        # I stored the tweet data in a database called 'training_tweets' in MongoDB, if
        # 'training_tweets' does not exist it will be created for you.
        db = client.training_tweets

        # Decode JSON
        datajson = json.loads(data)

        # I'm only storing tweets in English. I stored the data for these tweets in a collection
        # called 'training_tweets_collection' of the 'training_tweets' database. If
        # 'training_tweets_collection' does not exist it will be created for you.
        if "lang" in datajson and datajson["lang"] == "en":
            db.training_tweets_collection.insert_one(datajson)
            self.counter += 1

        if self.counter >= self.limit:
            stream.disconnect()



if __name__ == "__main__":
    # Since the account is private, the keys and secrets will not be shown here.
    # If any questions, please feel free to contact me.

    consumer_key = "*****Your Consumer Key*****"
    consumer_secret = "*****Your Consumer Secret*****"
    access_token_key = "*****Your Access Token Key*****"
    access_token_secret = "*****Access Token Secret*****"

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token_key, access_token_secret)

    # LOCATIONS are the longitude, latitude coordinate corners for a box that restricts the
    # geographic area from which you will stream tweets. The first two define the southwest
    # corner of the box and the second two define the northeast corner of the box.
    LOCATIONS = [-124.7771694, 24.520833, -66.947028, 49.384472,  # Contiguous US
                 -164.639405, 58.806859, -144.152365, 71.76871,  # Alaska
                 -160.161542, 18.776344, -154.641396, 22.878623]  # Hawaii

    stream_listener = StreamListener(api=tweepy.API)
    stream = tweepy.Stream(auth=auth, listener=stream_listener,proxy="127.0.0.1:1080", wait_on_rate_limit=True,
                           wait_on_rate_limit_notify=True)
    stream.filter(locations=LOCATIONS)