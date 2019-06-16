# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-12
@environment : python 3.6
@author      : zhangjiwen
@file        : test.py
"""

import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import Graph
from multiprocessing import cpu_count
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# In[0]:
def build_graph_fromcsv(csv_path, isnode = False, isedge = False):
    '''
    build directed graph_from csv_file
    :param csv_path: file to read
    :param isnode: if True, add nodes in fraph
    :param isedge: if True, add edges in fraph
    :return: Digraph
    '''
    dt = pd.read_csv(csv_path, encoding='utf-8')
    g = nx.DiGraph()
    if isnode:
        nodes = dt['user_id']
        nodes = list(map(int, nodes))
        g.add_nodes_from(nodes)
    elif isedge:
        edges = []
        for i in range(0, len(dt['from'])):
            edges.append( (int(dt['from'][i]), int(dt['to'][i])) )
        g.add_edges_from(edges)

    g.name = 'users_relation_graph'
    print(nx.info(g), "\n---------------------------------------\n")

    return g


# In[1]:
def build_corpus(G, max_paths, path_len, save_walks):
    '''
    Generating random walks
    :param G: networkx graph
    :param max_paths: num to generate a randomwalk in a node
    :param path_len: length of each path
    :param save_walks: save walks to txt or not
    :return: deep walk path list
    '''

    print("\t**Stage 1 : Generating random walks**")

    # Build corpus based on random walks
    corpus_cur = Graph.build_walk_corpus(G=G, max_paths=max_paths, path_len=path_len)

    #print("\nNumber of walks in the corpus = ", len(corpus_cur))

    if save_walks:
        Graph.save_corpus(max_paths, path_len, corpus)

    return corpus_cur


# In[2]:
def generate_embeddings(d, w, hs, corpus, save_emb):
    '''
    Train model
    :param d: dimension of nodes
    :param w: window size
    :param hs: use hierarchical softmax or not
    :param corpus: deepwalk paths
    :param save_emb: save embeddings to txt
    :return:
    '''
    # TODO: try negative sampling (hs=0)

    print("\t**Stage 2 : Generating Embeddings for nodes using Word2Vec**")
    print("Word2Vec parameters : Dimensions = " + str(d) + ", window = " + str(w) + ", hs = " + str(
        hs) + ", number of cpu cores assigned for training = " + str(cpu_count()))

    model = Word2Vec(size=d, window=w, sg=1, min_count=0, hs=hs, workers=cpu_count())
    model.build_vocab(corpus)
    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)

    print("Model training done. Word2Vec embeddings generated.")

    word_vec = model.wv

    if save_emb:
        # Save w2v embeddings
        name = 'word2vec-d' + str(d) + '-w' + str(w) + '-hs' + str(hs) + '.txt'
        word_vec.save_word2vec_format(binary=False, fname=name)
        print("Embeddings saved to file -> ", name)

    return word_vec


# In[3]:
def initialize_feature(g):
    '''
    Use Random walk to generate embdeddings
    :param g: graph
    :return: word_vec
    '''

    G = g

    len = 30            # 请输入随机游走的最长步数
    walks = 10          # 请输入每个结点发起随机游走的次数
    w = False           # 请输入是否需要额外存储随机游走信息(True/False)
    dimensions = 128    # Dimensions of word embeddings
    window = 5          # Window size for skipgram
    hs = 1              # 0 - Negative Sampling  1 - Hierarchical Softmax
    save_emb = 1        # Flag to save word embeddings to disk

    # 利用随机游走建立语料库
    corpus = build_corpus(G, max_paths = walks, path_len = len, save_walks = w)

    # 产生embedding
    word_vec = generate_embeddings(dimensions, window, hs, corpus, save_emb)

    return word_vec


# In[4]:
def load_embeddings(path):
    data = KeyedVectors.load_word2vec_format(fname=path, binary=False)
    # data.vocab : dict of nodes and its embeddings
    nodes = list(data.vocab.keys())
    edges_np = []
    c = 0
    for node in nodes:
        vec = data[node]
        edges_np.append(vec)

    edges_np = np.asarray(edges_np)

    return nodes, edges_np


# In[5]:
def naive_draw_graph(G = nx.generators.directed.random_k_out_graph(10, 3, 0.5)):
    print("***** now prepare for drawing Graph *****")
    print("\t layouting")
    #pos = nx.layout.spring_layout(G)
    pos = nx.layout.spectral_layout(G)
    print("\t drawing")
    node_sizes = [ len(list(G.neighbors(node)))*50 for node in G.nodes()]

    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    edges = nx.draw_networkx_edges(G, pos, node_size=node_sizes, arrowstyle='->',
                                   arrowsize=10, edge_color=edge_colors,
                                   edge_cmap=plt.cm.Blues, width=2)
    pc = mpl.collections.PatchCollection(edges, cmap=plt.cm.Blues)
    pc.set_array(edge_colors)
    plt.colorbar(pc)

    plt.show()
    print("***** End drawing Graph *****")


# In[6]:
def PCA(X_train, k):

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


# In[7]:
def draw_graph(G, pos):
    print("***** now drawing Graph *****")

    node_sizes = [ len(list(G.neighbors(node))) for node in G.nodes()]

    M = G.number_of_edges()
    edge_colors = range(2, M + 2)
    d_nodes = nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='blue')
    #d_edges = nx.draw_networkx_edges(G, pos, arrowstyle='->',
    #                               arrowsize=10, edge_color=edge_colors,
    #                               edge_cmap=plt.cm.Blues, width=2)
    #pc = mpl.collections.PatchCollection(d_edges, cmap=plt.cm.Blues)
    #pc.set_array(edge_colors)
    #plt.colorbar(pc)

    plt.show()
    print("***** End drawing Graph *****")


# In[8]:
if __name__ == "__main__":

    path = 'data/graph_data/edges2.csv'
    g = build_graph_fromcsv(path, isedge=True)

    has_emb = 0  # Already Has Embedding or NOT (0/1)
    emb_file = 'data/word2vec-d128-w5-hs1.txt'  # file_path

    if not has_emb:
        word_vec = initialize_feature(g)
    else:
        Nodes, edges_np = load_embeddings(emb_file)

        # TODO: PCA -  Draw graph
        # naive_draw_graph() : very very slow to compute layout
        # but u can use it to play
        # for large scale plotting, choose PCA instead
        eig_vec = PCA(edges_np, 2)
        print("eigen vectors shape : ",eig_vec.shape)
        pos = {}

        for node in g.nodes():
            i = Nodes.index(str(node))
            pos[node] = list(eig_vec[i])

        draw_graph(g, pos)
