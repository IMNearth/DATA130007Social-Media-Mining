# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-16 14:22
@environment : python 3.6
@author      : zhangjiwen
@file        : 5_create_meta2vec_data.py

Note that this random walk method is generally based on
    metapath2vec: Scalable Representation Learning for Heterogeneous Networks(KDD 2017)
And this file is purposed to prepare corpus for metapath2vec++.
"""

import pandas as pd
import networkx as nx
import unicodecsv as csv
import random
import re
import numpy as np



# In[0]:
class MetaWalkClient():
    def __init__(self, meta_path):
        '''
        :param meta_path: during random walk, the node type u want to choose
                            e.g. meta_path = ['v','i'] means v->i->v->i->v->...
                            e.g. meta_path = ['i','i','v'] means i->i->v->i->i->v->...
        '''
        self.meta_path = meta_path
        self.graph = nx.Graph()

    def load_graph_fromcsv(self, csv_path, is_geo_node=False, is_user_node=False, isedge=False):
        '''
        build directed graph_from csv_file
        can also be used to add data in graph, u can use it for many times if u want

        :param csv_path: file to read
        :param is_geo_node: if True, add geo nodes in graph
        :param is_user_node: if True, add user nodes in graph
        :param isedge: if True, add edges in graph
        :return: Digraph

        Note that we mark on different type of nodes by the prefix of node_str.
        -- id_nodes start with 'i', e.g. 'i13569'
        -- geo_nodes start with 'v', e.g. 'vChina','vBeijing'
        Please make sure that your csv_nodes ans csv_edges has been dealt based on above principle.

        '''
        dt = pd.read_csv(csv_path, encoding='utf-8', dtype=str)
        g = self.graph
        if is_user_node:
            nodes = dt['i_user_id'] # 'i'+user_id
            g.add_nodes_from(nodes)
            # u can also use attributes to mark on this edge
            # e.g.  g.add_nodes_from(nodes, type = 'i-i')
        elif is_geo_node:
            nodes = dt['v_geo_state']  # 'v'+geo_state
            g.add_nodes_from(nodes)
        elif isedge:
            edges = []
            for i in range(0, len(dt['from'])):
                edges.append((dt['from'][i], dt['to'][i]))
            g.add_edges_from(edges)

        print(nx.info(g), "\n---------------------------------------\n")

        self.graph = g

    def random_walk(self, start_node, walklength):
        '''
        :param start_node: to begin with
        :param walklength : naively means how long the max length is
        :return: a line of str, different nodes are separated by space
                e.g. outline = "i12345 vChina i34567"
        '''
        g = self.graph
        mp = self.meta_path

        outline = "" + start_node
        current = start_node
        bias = mp.index(start_node[0])+1

        for i in range(0, walklength-1):
            neighbors = list(g.neighbors(current))
            if len(neighbors) == 0: break

            next_type = mp[(i + bias) % len(mp)]
            #print(current, neighbors, next_type)
            valid_neighbors = [n for n in neighbors if n[0]==next_type]
            if len(valid_neighbors) == 0: break

            current = random.choice(valid_neighbors)
            outline += " " + current

        return outline

    def generate_corpus(self, out_path, walklength, numWalks):
        '''
        :param out_path: path to a txt file
        :param walklength: length of each walk
        :param numWalks: number of walks per node
        '''
        g = self.graph

        with open(out_path, 'w') as f:
            for i in range(0, numWalks):
                for node in g.nodes():
                    # always start with type meta_path[0]
                    if node[0] != self.meta_path[0]: continue
                    outline = self.random_walk(node, walklength)
                    f.write(outline + "\n")

        print("***** Corpus Created : %s" % out_path)


# In[2]:
if __name__ == "__main__":
    # u can write your own test codes here
    # print("******** Step 1 : Create edges for Graph ********")

    # print("******** Step 2 : Generate corpus to be learned *******")

    meta_path = ['i', 'v', 'a', 'v','i']
    client = MetaWalkClient(meta_path)

    user_geo_edges = 'twitter/data/modified/group1_user2UScity.csv'
    user_edges = 'twitter/data/modified/has_US_tweeted_userFriends.csv'
    city_state = 'twitter/data/modified/group1_city2state2country.csv'

    client.load_graph_fromcsv(user_geo_edges, isedge=True)
    client.load_graph_fromcsv(user_edges, isedge=True)
    client.load_graph_fromcsv(city_state, isedge=True)

    out_path = 'twitter/data/modified/group1_corpus_ivav.txt'
    client.generate_corpus(out_path, 13, 30)


    # print("******** Step 3 : Use Terminal to run metapath2vec++ *******")

    #with open('/Users/zhangjiwen/Desktop/meta_learned.txt', 'r') as f:
    #    count = 0
    #    for lines in f:
    #        print(lines)
    #        count +=1
    #        if count >=100 : break
    #pass
