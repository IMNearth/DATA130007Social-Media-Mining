# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-13 10:22
@environment : python 3.6
@author  : zhangjiwen
@file    : 4_graph_analysis.py.py
"""

import pandas as pd
import networkx as nx
from networkx.algorithms import centrality
from pyecharts.charts import Line, Scatter
from pyecharts import options as opts
import numpy as np
import matplotlib.pyplot as plt
import unicodecsv as csv


class GraphAnalyser():
    def centrality_analysis(self, G, isDriected=False):
        '''
        :param g: Digraph()/ Graph()
        :return: several types of centrality of each nodes
        '''
        nodes = G.nodes()
        if isDriected:
            in_dc = centrality.in_degree_centrality(G)
            out_dc = centrality.out_degree_centrality(G)
            bc = centrality.betweenness_centrality(G)
            ec = centrality.eigenvector_centrality(G)

            cent = {}
            for node in nodes:
                cent[node] = [in_dc[node], out_dc[node], bc[node], ec[node]]
            print("Four types of centrality are calculated \n" +
                  "\n\tin_degree_centrality\n\tout_degree_centrality\n\tbetweenness_centrality\n\teigenvector_centrality")
            return cent
        else:
            dc = centrality.degree_centrality(G)
            bc = centrality.betweenness_centrality(G)
            ec = centrality.eigenvector_centrality(G)

            cent = {}
            for node in nodes:
                cent[node] = [dc[node], bc[node], ec[node]]
            print("Three types of centrality are calculated \n" +
                  "\n\tdegree_centrality\n\tbetweenness_centrality\n\teigenvector_centrality")
            return cent

    def degree_analysis(self, G):
        return nx.degree_histogram(G)


def draw_scatter_degree(degree_hist):
    x = range(0,len(degree_hist))
    #y = [z/float(sum(degree_hist))for z in degree_hist]
    #print(y)
    y = degree_hist

    c = (
        Scatter(init_opts=opts.InitOpts(page_title="Degree Distribution"))
            .add_xaxis(xaxis_data=list(x))
            .add_yaxis("degree frequency", y_axis=y, symbol_size=5)
            .set_global_opts(
                title_opts=opts.TitleOpts(title="Degree Distribution"),
                yaxis_opts=opts.AxisOpts(splitline_opts=opts.SplitLineOpts(is_show=True)),
        )
    )
    c.render("/Users/zhangjiwen/Desktop/0527_scatter_degree.html")


def mlt_degree(degree_hist):
    x = range(len(degree_hist))
    y = [np.log2(z) / float(sum(degree_hist)) for z in degree_hist]
    plt.scatter(x, y, c=y, linewidth=1, marker='o', alpha=0.5)
    plt.colorbar()
    plt.ylim(-0.0005, 0.0025)
    plt.grid(True)
    plt.title("Degree Distribution")
    plt.show()

#mlt_degree(degree_hist)


def draw_central(G, cent, pos = 0):
    '''
    nodelist = [item[0] for item in cent]
    node_central = [item[1]*10000 for item in cent]

    if not pos:
        pos = nx.spring_layout(G, iterations=200)

    nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
                           node_color=node_central,
                           node_size=50,
                           cmap=plt.cm.Blues)
    nx.draw_networkx_edges(G, pos, edge_color='grey')
    plt.show()

    return pos
    '''
    pass
#draw_central(g, degree_centrality)


def cmp_central(k, cent, centrality):
    order = np.argsort(centrality)[::-1]
    users = np.take(list(cent.keys()), order[:k])
    return list(order), list(users)


def main():
    g = nx.Graph()

    dt = pd.read_csv("/Users/zhangjiwen/Desktop/twitter/data/modified/1_has_US_tweeted_userFriends.csv",
                     encoding='utf-8')
    edges = []
    for i in dt.index:
        edges.append(
            (dt['from'][i], dt['to'][i])
        )
    g.add_edges_from(edges)
    print(nx.info(g), "\n---------------------------------------\n")

    worker = GraphAnalyser()
    cent = worker.centrality_analysis(g)
    degree_hist = worker.degree_analysis(g)

    return g, degree_hist, cent

if __name__ == "__main__":
    g, degree_hist, cent = main()
    mlt_degree(degree_hist)

    degree_centrality = []
    betweenness_centrality = []
    eigenvector_centrality = []

    for user, c in cent.items():
        degree_centrality.append(c[0])
        betweenness_centrality.append(c[1])
        eigenvector_centrality.append(c[2])

    # pos = draw_central(g, degree_centrality)

    od1, dc_users = cmp_central(100, cent, degree_centrality)
    od2, bc_users = cmp_central(100, cent, betweenness_centrality)
    od3, ec_users = cmp_central(100, cent, eigenvector_centrality)

    save = 1
    if save:
        path = '/Users/zhangjiwen/Desktop/twitter/data/modified/Centrality.csv'
        f = csv.writer(open(path, "wb+"))
        f.writerow(['order', 'user', 'degree_centrality','user', 'betweenness_centrality', 'user', 'eigenvector_centrality'])

        users = list(cent.keys())

        for i in range(0, len(od1)):
            f.writerow([i+1,
                        users[od1[i]], degree_centrality[od1[i]],
                        users[od2[i]], betweenness_centrality[od2[i]],
                        users[od3[i]], eigenvector_centrality[od3[i]],
                        ])

        print("save to file --> %s" % path)

    '''
    save = 1
    if save:
        path = '/Users/zhangjiwen/Desktop/twitter/data/modified/Centrality_comparison.csv'
        f = csv.writer(open(path, "wb+"))
        f.writerow(['degree_centrality', 'betweenness_centrality', 'eigenvector_centrality'])

        for i in range(0, 100):
            f.writerow([dc_users[i], bc_users[i], ec_users[i]])

        print("save to file --> %s" % path)
    '''


