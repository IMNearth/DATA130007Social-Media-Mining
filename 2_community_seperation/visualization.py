# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-19 17:24
@environment : python 3.6
@author      : zhangjiwen
@file        : 0518_test.py
"""
import pandas as pd
from pyecharts.charts import Graph
from pyecharts import options as opts
from collections import Counter
from math import log
from time import strftime, localtime


def test():
    nodes = []
    cat = set()
    df = pd.read_csv('data/modified/3_group1_fuoutcome_user2city.csv')
    counts = Counter(list(df['community']))
    scarce_cat = [ c for c,__ in counts.items() if __<=10]
    for i in df.index:
        if df['community'][i] in scarce_cat:
            continue
        if df['user'][i][0] == 'i':
            cat.add(df['community'][i])
            if int(df['degree'][i]) > 10:
                nodes.append({
                    "name": df['user'][i][1:],
                    "category": int(df['community'][i]),
                    "symbolSize": 10 + 0.5*int(df['degree'][i]),
                })
            elif int(df['degree'][i]) > 100:
                nodes.append({
                    "name": df['user'][i][1:],
                    "category": int(df['community'][i]),
                    "symbolSize": 12 + log(int(df['degree'][i]), 2),
                })
            else:
                nodes.append({
                    "name": df['user'][i][1:],
                    "category": int(df['community'][i]),
                    "symbolSize": 10 + int(df['degree'][i]),
                })
        elif df['user'][i][0] == 'v':
            cat.add(df['community'][i])
            nodes.append({
                "name": df['user'][i][1:],
                "category": int(df['community'][i]),
                "symbolSize": 10 + 0.2*int(df['degree'][i]),
            })

    category = []
    for c in cat:
        category.append({
            "name": str(c)
        })
    print("\t category", len(category))

    edges = []

    '''
    df = pd.read_csv('data/modified/1_has_US_tweeted_userFriends.csv')
    for i in df.index:
        edges.append({
            "source": df['from'][i][1:],
            "target": df['to'][i][1:]
        })
    '''

    df = pd.read_csv('data/modified/1_group1_user2UScity.csv')
    for i in df.index:
        edges.append({
            "source": df['from'][i][1:],
            "target": df['to'][i][1:]
        })
    return nodes, edges, category


def Force(nodes, edges, category):
    print(len(nodes))
    print(len(edges))
    graph = (
        Graph(init_opts=opts.InitOpts(width='1600px', height='900px', page_title="Force Induced Graph"))
            .add("", nodes, edges, category, layout='force', repulsion=50, gravity=0.5,
                 linestyle_opts=opts.LineStyleOpts(curve=0.05),
                 label_opts=opts.LabelOpts(is_show=False)
                 )
            .set_global_opts(
            legend_opts=opts.LegendOpts(is_show=True, pos_left='10%', selected_mode='multiple', orient='vertical'),
            title_opts=opts.TitleOpts(title="Graph_"+strftime("%m-%d", localtime())))
    )
    time_str = strftime("%m-%d", localtime())
    graph.render("Desktop/Graph"+ time_str +".html") #


if __name__ == '__main__':
    nodes, edges, category = test()
    Force(nodes, edges, category)
