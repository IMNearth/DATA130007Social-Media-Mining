# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-10
@environment : python 3.6
@author      : zhangjiwen
@file        : test.py
"""

import pandas as pd
import unicodecsv as csv
import re


def read_csv_file(path, columns):
    '''
    read columns in csv
    :param path:
    :param columns:
    :return: list of column context
    '''
    dt = pd.read_csv(path, encoding='utf-8')
    ans = []
    for col in columns:
        ans.append(dt[col])
    return ans

def save_2_csv(dt, columns, path):
    '''
    save data to csv file
    :param dt: list of column data
    :param columns: column names
    :param path: save path
    '''
    c = 0
    with open(path, "wb+") as file:
        f = csv.writer(file)

        # Column names
        f.writerow(columns)

        for i in range(0, len(dt[0])):
            f.writerow( [item[i] for item in dt] )
            c += 1
            # Status update
            if c % 1000 == 0:
                print("Just stored data %d" % c)


def create_nodes(open_path, save_path):
    columns = ['user_id', ]
    dt = read_csv_file(open_path, columns)
    save_2_csv(dt, columns, save_path)


def create_edges(open_path, save_path):
    columns = ['user_id', 'followers_id', 'friends_id']
    dt = read_csv_file(open_path, columns)

    src = []
    dst = []
    for i in range(0, len(dt[0])):
        user = dt[0][i]

        if type(dt[1][i]) == str:
            for follower in re.findall(r"[0-9]+", dt[1][i]):
                if follower != '0':
                    src.append(follower)
                    dst.append(user)

        if type(dt[2][i]) == str:
            for friend in re.findall(r"[0-9]+", dt[2][i]):
                if friend != '0':
                    src.append(user)
                    dst.append(friend)
                    #src.append(friend)
                    #dst.append(user)

    print(len(src), "\t", len(src)==len(dst))
    relation = [src, dst] # src -> dst
    save_2_csv(relation, ['from', 'to'], save_path)


if __name__ == "__main__":

    open_path = '/Users/zhangjiwen/Desktop/twitter/data/unique_sorted_counted_userid.csv'
    save_path = '/Users/zhangjiwen/Desktop/twitter/data/graph_data/nodes.csv'
    #create_nodes(open_path, save_path)


    open_path = '/Users/zhangjiwen/Desktop/twitter/data/userscl_new2_final.csv'
    save_path = '/Users/zhangjiwen/Desktop/twitter/data/graph_data/edges_2.csv'
    create_edges(open_path, save_path)
