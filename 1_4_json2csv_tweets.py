# -*- coding: utf-8 -*-
"""
@author: Yin Wan
@environment: Python 3.7
@file: json2csv_tweets.py
@time: 2019/5/8 
"""
import json
import unicodecsv as csv

def tweets_json_to_csv(file_list, tweets_output_file,users_output_file):
    '''

    :param file_list: A list of JSON files
    :param tweets_output_file: tweets.csv
    :param users_output_file: users.csv
    :return:
    '''

    count = 0
    f_tweets = csv.writer(open(tweets_output_file, "wb+"))
    f_users = csv.writer(open(users_output_file, "wb+"))
    # Column names
    f_tweets.writerow(['user_id',
                       'tweet',  # relabelled: the API calls this 'text'
                       'country_code',
                       'tweet_geo_location',  # relabelled: the API calls this 'full_name'
                       'tweet_bounding_box',
                       'tweet_created_at'])
    f_users.writerow(['user_id',
                      'user_screen_name',
                      'user_geo_location',
                      'user_created_at'])
    for file_ in file_list:
        with open(file_, "r",encoding='utf-8') as r:
            for line in r:
                try:
                    tweet = json.loads(line)
                except:
                    continue
                if tweet and tweet['place'] != None and tweet['place']['bounding_box'] != None:
                    f_tweets.writerow([tweet['user']['id_str'],
                                       tweet['text'],
                                       tweet['place']['country_code'],
                                       tweet['place']['full_name'],
                                       tweet['place']['bounding_box']['coordinates'],
                                       tweet['created_at']])
                    f_users.writerow([tweet['user']['id_str'],
                                      tweet['user']['screen_name'],
                                      tweet['user']['location'],
                                      tweet['user']['created_at']])
                    count += 1

                    # Status update
                    if count % 100000 == 0:
                        print('Just stored tweet #{}'.format(count))


if __name__ == "__main__":
    tweets_json_to_csv(['training_user_tweets.json'], 'training_user_tweets1_3_tweets2.csv',
                       'training_user_tweets1_3_users2.csv')