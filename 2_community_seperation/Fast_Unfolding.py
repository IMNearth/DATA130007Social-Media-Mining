# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-19 21:32
@environment : python 3.6
@author      : zhangjiwen
@file        : 0518_test.py
"""

import networkx as nx
import pandas as pd
from itertools import permutations
from itertools import combinations
from collections import defaultdict
import numpy as np
import unicodecsv as csv


class FastUnfolding(object):
    def __init__(self):
        self.MIN_VALUE = 0.0000001
        self.node_weights = {}    #节点权重

    @classmethod
    def convertIGraphToNxGraph(cls, igraph):
        node_names = igraph.vs["name"]
        edge_list = igraph.get_edgelist()
        weight_list = igraph.es["weight"]
        node_dict = defaultdict(str)

        for idx, node in enumerate(igraph.vs):
            node_dict[node.index] = node_names[idx]

        convert_list = []
        for idx in range(len(edge_list)):
            edge = edge_list[idx]
            new_edge = (node_dict[edge[0]], node_dict[edge[1]], weight_list[idx])
            convert_list.append(new_edge)

        convert_graph = nx.Graph()
        convert_graph.add_weighted_edges_from(convert_list)
        return convert_graph

    def updateNodeWeights(self, edge_weights):
        node_weights = defaultdict(float)
        for node in edge_weights.keys():
            node_weights[node] = sum([weight for weight in edge_weights[node].values()])
        return node_weights

    def getBestPartition(self, graph, param=1.):
        node2com, edge_weights = self._setNode2Com(graph)    #获取节点和边

        node2com = self._runFirstPhase(node2com, edge_weights, param)
        best_modularity = self.computeModularity(node2com, edge_weights, param)

        partition = node2com.copy()
        new_node2com, new_edge_weights = self._runSecondPhase(node2com, edge_weights)

        while True:
            new_node2com = self._runFirstPhase(new_node2com, new_edge_weights, param)
            modularity = self.computeModularity(new_node2com, new_edge_weights, param)
            if abs(best_modularity - modularity) < self.MIN_VALUE:
                break
            best_modularity = modularity
            partition = self._updatePartition(new_node2com, partition)
            _new_node2com, _new_edge_weights = self._runSecondPhase(new_node2com, new_edge_weights)
            new_node2com = _new_node2com
            new_edge_weights = _new_edge_weights

        return partition

    def computeModularity(self, node2com, edge_weights, param):
        q = 0
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2

        com2node = defaultdict(list)
        for node, com_id in node2com.items():
            com2node[com_id].append(node)

        for com_id, nodes in com2node.items():
            node_combinations = list(combinations(nodes, 2)) + [(node, node) for node in nodes]
            cluster_weight = sum([edge_weights[node_pair[0]][node_pair[1]] for node_pair in node_combinations])
            tot = self.getDegreeOfCluster(nodes, node2com, edge_weights)
            q += (cluster_weight / (2 * all_edge_weights)) - param * ((tot / (2 * all_edge_weights)) ** 2)
        return q

    def getDegreeOfCluster(self, nodes, node2com, edge_weights):
        weight = sum([sum(list(edge_weights[n].values())) for n in nodes])
        return weight

    def _updatePartition(self, new_node2com, partition):
        reverse_partition = defaultdict(list)
        for node, com_id in partition.items():
            reverse_partition[com_id].append(node)

        for old_com_id, new_com_id in new_node2com.items():
            for old_com in reverse_partition[old_com_id]:
                partition[old_com] = new_com_id
        return partition

    def _runFirstPhase(self, node2com, edge_weights, param):
        # 计算所有边上的权重之和
        all_edge_weights = sum(
            [weight for start in edge_weights.keys() for end, weight in edge_weights[start].items()]) / 2
        self.node_weights = self.updateNodeWeights(edge_weights) #输出一个字典，每个node对应node上边的权重和
        status = True
        while status:
            statuses = []
            for node in node2com.keys():   # 逐一选择节点和周边连接的节点进行比较
                #statuses = []
                com_id = node2com[node]    # 获取节点对应的社团编号
                neigh_nodes = [edge[0] for edge in self.getNeighborNodes(node, edge_weights)] #获取连接的所有边节点

                max_delta = 0.              # 用于计算比对
                max_com_id = com_id         # 默认当前社团id为最大社团id
                communities = {}
                for neigh_node in neigh_nodes:
                    node2com_copy = node2com.copy()
                    if node2com_copy[neigh_node] in communities:
                        continue
                    communities[node2com_copy[neigh_node]] = 1
                    node2com_copy[node] = node2com_copy[neigh_node] # 把node对应的社团id放到临近的neigh_node中

                    delta_q = 2 * self.getNodeWeightInCluster(node, node2com_copy, edge_weights) - (self.getTotWeight(
                        node, node2com_copy, edge_weights) * self.node_weights[node] / all_edge_weights) * param
                    if delta_q > max_delta:
                        max_delta = delta_q                     # max_delta 选择最大的增益的node
                        max_com_id = node2com_copy[neigh_node]  # 对应 max_com_id 选择最大的增益的临接node的id

                node2com[node] = max_com_id
                statuses.append(com_id != max_com_id)

            print(sum(statuses))

            if sum(statuses) == 0:
                break

        return node2com

    def _runSecondPhase(self, node2com, edge_weights):
        """
        :param node2com:       第一层phase 构建完之后的node->社团结果
        :param edge_weights:   社团边字典
        :return:
        """
        com2node = defaultdict(list)

        new_node2com = {}
        new_edge_weights = defaultdict(lambda: defaultdict(float))

        for node, com_id in node2com.items():
            #生成了社团：--->节点映射
            com2node[com_id].append(node)  #添加同一一个社团id对应的node
            if com_id not in new_node2com:
                new_node2com[com_id] = com_id

        nodes = list(node2com.keys())
        node_pairs = list(permutations(nodes, 2)) + [(node, node) for node in nodes]
        for edge in node_pairs:
            new_edge_weights[new_node2com[node2com[edge[0]]]][new_node2com[node2com[edge[1]]]] += edge_weights[edge[0]][
                edge[1]]
        return new_node2com, new_edge_weights

    def getTotWeight(self, node, node2com, edge_weights):
        """
        :param node:
        :param node2com:
        :param edge_weights:
        :return:
        """
        nodes = [n for n, com_id in node2com.items() if com_id == node2com[node] and node != n]

        weight = 0.
        for n in nodes:
            weight += sum(list(edge_weights[n].values()))
        return weight

    def getNeighborNodes(self, node, edge_weights):
        """
        :param node:  输入节点
        :param edge_weights: 边字典
        :return: 输出每个节点连接点边集合
        """
        if node not in edge_weights:
            return 0
        return edge_weights[node].items()

    def getNodeWeightInCluster(self, node, node2com, edge_weights):
        neigh_nodes = self.getNeighborNodes(node, edge_weights)
        node_com = node2com[node]
        weights = 0.
        for neigh_node in neigh_nodes:
            if node_com == node2com[neigh_node[0]]:
                weights += neigh_node[1]
        return weights

    def _setNode2Com(self,graph):
        """
        :return: 节点->团，edge_weights 形式：{'a': defaultdict(<class 'float'>, {'c': 1.0, 'b': 1.0})}
        """
        node2com = {}
        edge_weights = defaultdict(lambda: defaultdict(float))
        for idx,node in enumerate(graph.nodes()):
            node2com[node] = idx    #给每一个节点初始化赋值一个团id
            for edge in graph[node].items():
                edge_weights[node][edge[0]] = edge[1]['weight']
        return node2com,edge_weights


if __name__ == "__main__":
    # build edges
    # note that we only use the user-UScity information and user-friends information
    edges = []

    dt = pd.read_csv("/Users/zhangjiwen/Desktop/twitter/data/modified/1_group1_user2UScity.csv")
    for i in dt.index:
        edges.append(
            (dt['from'][i],dt['to'][i] )
        )

    '''
    dt = pd.read_csv("/Users/zhangjiwen/Desktop/twitter/data/modified/1_has_US_tweeted_userFriends.csv")
    for i in dt.index:
        edges.append(
            (dt['from'][i],dt['to'][i] )
        )
    '''

    dt = pd.read_csv('/Users/zhangjiwen/Desktop/twitter/data/modified/1_group1_city2state2country.csv')
    for i in dt.index:
        if str(dt['from'][i]) != 'US' and str(dt['to'][i]) != 'US':
            edges.append(
                (dt['from'][i], dt['to'][i])
            )

    print(len(edges))
    print("build weight...")
    path = '/Users/zhangjiwen/Desktop/twitter/data/modified/2_group1_corpus_ivav_learned_0520.txt'
    dt = np.loadtxt(path, dtype=str, skiprows=2)
    userAndgeo = list(dt[:, 0])
    dt = np.delete(dt, 0, axis=1)
    dt = dt.astype(float)

    weight_edges = []
    for item1,item2 in edges:
        if item1 in userAndgeo and item2 in userAndgeo:
            i = userAndgeo.index(item1)
            j = userAndgeo.index(item2)
            distance = np.linalg.norm(dt[i,:]-dt[j,:], ord=2)
            #print("distance:", distance)
            if distance != 0:
                weight_edges.append((item1, item2, {'weight': 1.0 / distance}))
            #else:
            #    weight_edges.append((item1, item2, {'weight': 1e-2}))
        #else:
        #    weight_edges.append((item1, item2, {'weight': 0}))

    print(len(weight_edges))
    g = nx.Graph()
    g.add_edges_from(weight_edges)
    print(nx.info(g), "\n---------------------------------------\n")

    delete_node = []
    addback_edge = []
    for node in g:
        if g.degree[node] == 1:
            delete_node.append(node)
            for nbr in g.neighbors(node):
                addback_edge.append( (node, nbr) )

    print(len(delete_node))
    g.remove_nodes_from(delete_node)
    print(nx.info(g), "\n---------------------------------------\n")


    print("now entering fast-unfolding")
    client = FastUnfolding()
    partition = client.getBestPartition(g)

    print("now deal with index.. ")
    p = defaultdict(list)
    classes = set()
    for node, com_id in partition.items():
        p[com_id].append(node)
        classes.add(com_id)

    g.add_edges_from(addback_edge)

    for node, nbr in addback_edge:
        com_id = partition[nbr]
        p[com_id].append(node)
    
    classes = list(classes)
    print("classes : ", len(classes))

    c = 0
    with open("/Users/zhangjiwen/Desktop/twitter/data/modified/group1_fuoutcome_user2city.csv",  "wb+") as file:
        f = csv.writer(file)
        f.writerow(['user', 'community', 'degree'])
        for community, iTems in p.items():
            for sth in iTems:
                f.writerow([str(sth), str(classes.index(community)+1), len(list(g.neighbors(sth)))])
                c += 1
                # Status update
                if c % 1000 == 0:
                    print("Just stored data %d" % c)

