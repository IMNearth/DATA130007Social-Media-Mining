# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-14 23:22
@environment : python 3.6
@author      : zhangjiwen
@file        : semantic_analysis.py
"""

import requests
# pprint is pretty print (formats the JSON)
from pprint import pprint
import pandas as pd
import unicodecsv as csv
import time


def get_documents(path):
    '''
    :param path: path to csv file
    :return: a document-type object
    '''
    dt = pd.read_csv(path, encoding='utf-8')
    ans = []
    count = 0
    for i in range(0, len(dt['user_id'])):
        k = int(count / 1000)
        if count % 1000 == 0:
            ans.append({'documents': []})
        ans[k]['documents'].append({
                'id': str(i),
                'language': 'en',
                'text': dt['tweet'][i]
        })
        count +=1
        #if count >= 10000: break

    j = 0
    for item in ans:
        j+=1
        print("\t %d  %d tweets to be analysed" % (j, len(item['documents'])))
    #pprint(ans)
    return ans, dt, count


def save_sentiments_2_csv(out_csv_path, sentiments, documents, dt, ceiling):
    c = 0
    columns = ['num', 'user_id', 'emotion(0-1)', 'country_code',
               'tweet_geo_city', 'tweet_geo_state', 'tweet', 'goodtime']

    with open(out_csv_path, "wb+") as file:
        f = csv.writer(file)

        # Column names
        f.writerow(columns)
        for k in range(0, len(sentiments)):
            senti_list = sentiments[k]['documents']
            tweet_list = documents[k]['documents']

            for i in range(0, len(senti_list)):
                num = tweet_list[i]['id']
                usr_id = dt['user_id'][c]
                score = senti_list[i]['score']
                country_code = dt['country_code'][c]
                city = dt['tweet_geo_city'][c]
                state = dt['tweet_geo_state'][c]
                tweet = tweet_list[i]['text']
                time = dt['goodtime'][c]
                f.writerow([num, usr_id, str(score), country_code, city, state, tweet, time])
                c += 1
            # Status update
                if c % 1000 == 0:
                    print("Just stored data %d" % c)
                if c >= ceiling:
                    break
    pass


if __name__ == "__main__":
    subscription_key = '91b6361ff7d84fa18e8dd45f40a61a78' #key1
    # key2 c40f328a58e44890aa2032e2f500aa36
    assert subscription_key

    text_analytics_base_url = "https://westcentralus.api.cognitive.microsoft.com/text/analytics/v2.1/"
    sentiment_api_url = text_analytics_base_url + "sentiment"
    #print(sentiment_api_url)

    csv_file_path = '/Users/zhangjiwen/Desktop/twitter/data/raw/group1_fulltweets.csv'
    document_list, dt, ceiling = get_documents(csv_file_path)
    print("document:", len(document_list))

    '''
    document-type objects
    documents = {'documents': [
        {'id': '1', 'language': 'en',
         'text': 'I had a wonderful experience! The rooms were wonderful and the staff was helpful.'},
        {'id': '2', 'language': 'en',
         'text': 'I had a terrible time at the hotel. The staff was rude and the food was awful.'},
        {'id': '3', 'language': 'es',
         'text': 'Los caminos que llevan hasta Monte Rainier son espectaculares y hermosos.'},
        {'id': '4', 'language': 'es', 'text': 'La carretera estaba atascada. Había mucho tráfico el día de ayer.'}
    ]}
    '''

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    sentiment_list = []
    c = 0
    for documents in document_list:
        response = requests.post(sentiment_api_url, headers=headers, json=documents)
        sentiments = response.json()
        sentiment_list.append(sentiments)
        c += 1
        if c%10 == 0:
            time.sleep(10)
            print("\t now %d-th sleep" % int(c/10000))
        #pprint(sentiments)

    print("sentiment", len(sentiment_list))

    out_csv_path = '/Users/zhangjiwen/Desktop/twitter/data/test_sentiment.csv'
    save_sentiments_2_csv(out_csv_path, sentiment_list, document_list, dt, ceiling)
