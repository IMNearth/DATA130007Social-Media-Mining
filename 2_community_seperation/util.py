# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-19 16:55
@environment : python 3.6
@author      : zhangjiwen
@file        : util.py
"""
from networkx.algorithms import centrality
import numpy as np
import unicodecsv as csv
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from itertools import permutations
from itertools import combinations
from collections import defaultdict
from time import time, strftime, localtime

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
        dt = pd.read_csv(csv_path, encoding='utf-8')
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
                statuses = []
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


def centrality_analysis(G, isDriected = False):
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
        print("Four types of centrality are calculated \n"+
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



def k_means(embed_dict, classes, iter = 10, save = True):
    '''
    k - means clustering
    :param embed_dict: 每个结点的embedding向量，dict{node : vector}
    :param emb_size: embedding向量的维数
    :param classes: 类别划分
    :param iter: 迭代次数，default = 10
    :param save: 是否保存，default = True
    :param vocab_size: 结点数量
    :return: class_list -- cl
    '''
    nodes = list(embed_dict.keys())
    emb_vec = list(embed_dict.values())

    vocab_size = len(nodes)
    emb_size = len(emb_vec[0])

    centcn = [1 for __ in range(0, classes)]    # 每个中心点拥有的词数量
    cl = [0 for __ in range(0,vocab_size)]     # 每个词所属类别标签
    cent = np.zeros([classes, emb_size])       # 聚类中心，每个中心需要通过偏移量来定位

    # 初始化每个词所属类别
    for a in range(0, vocab_size):
        cl[a] = a % classes
    #print(cl)

    # start training
    for i in range(0, iter):
        # 求每个中心点每个维度值的总和，等于所有属于这个类别的词向量的相应维度相加
        for node_id in range(0, vocab_size):
            cent[cl[node_id]] += emb_vec[node_id]
            centcn[cl[node_id]] += 1

        # 对于每一个类别，需要更新中心点各维度值，就是总和平均
        for j in range(0, classes):
            cent[j] = cent[j]/centcn[j]
            norm2 = np.linalg.norm(cent[j], ord=2)
            # x_norm=np.linalg.norm(x, ord=None, axis=None, keepdims=False)
            # x: 表示矩阵; ord：范数类型, default = 2;
            # axis：处理类型  axis=1表示按行向量处理，求多个行向量的范数
            # keepding：是否保持矩阵的二维特性; True表示保持矩阵的二维特性，False相反
            cent[j] = cent[j]/ norm2

        # 更新每个词所属的类别，看离哪个中心点最近就归为相应的类别
        for k in range(0, vocab_size):
            distance = -9999
            nearest = 0
            for m in range(0, classes):
                t_dist = np.dot(emb_vec[k], cent[m])
                if t_dist > distance:
                    distance = t_dist
                    nearest = m
            cl[k] = nearest

    # print(cent)

    # Save the K-means classes
    if save:
        count = 0
        save_path = strftime("%m-%d-%H%M", localtime())+'k_means_outcome.csv'
        f = csv.writer(open(save_path, "wb+"))
        # Column names
        f.writerow(['user_id', 'classes'])

        for i in range(0, vocab_size):
            f.writerow([str(nodes[i]), str(cl[i])])
            count += 1

            # Status update
            if count % 1000 == 0:
                print('Just stored  #{}'.format(count))

        print("Save to ---> %s" % save_path)

    return cl


def PCA(X_train, k):
    '''
    project row vector in X_train to a lower k-dimension
    :param X_train: input, line vectors will be projected
    :param k: dimension of the space u want to project data on
    :return: projected data, ndarray
    '''
    print("***** start PCA *****")
    # Centerize the images
    X_train -= np.mean(X_train, axis=0)

    print('\tCalculating Covariance matrix')
    CovM = np.cov(X_train.T)

    print('\tCalculating eigen values and eigen vectors, please wait...')
    evals, evecs = np.linalg.eigh(CovM)
    # Sort the eigen values in descending order and then sorted the eigen vectors by the same index
    idx = np.argsort(evals)[::-1][:k]
    evecs = evecs[:, idx]

    # Can uncomment for plotting eigen values graph
    # evals = evals[idx]
    # pyplot.plot(evals)
    # pyplot.show()
    print("***** End PCA *****")
    return np.dot(evecs.T, X_train.T).T


def LDA(X_train, Y_train, X):
    '''
    Linear Discriminant Analysis
    :param X_train: ndarray
    :param Y_train: a vector
    '''
    clf = LinearDiscriminantAnalysis(n_components=2)
    clf.fit(X_train, Y_train)
    X_r2 = clf.transform(X)
    return X_r2


def draw_graph(G, pos, nodes, n_color = 'blue', all = 1):
    print("***** now drawing Graph *****")
    if all:
        nx.draw_networkx_nodes(G, pos, node_size=50)
    print("\t drawing nodes ...")
    nx.draw_networkx_nodes(G, pos,
                           nodelist=nodes,
                           node_size=50,
                           node_color=n_color)
    print("\t drawing edges ...")
    nx.draw_networkx_edges(G, pos, edge_color='grey')

    plt.show()
    print("***** End drawing Graph *****")

