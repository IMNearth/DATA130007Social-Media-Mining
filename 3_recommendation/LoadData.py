# -*- coding: utf-8 -*-
"""
@create_time : 2019-05-22 15:06
@environment : python 3.6
@author      : zhangjiwen
@file        : nfmLoadData.py
"""
import numpy as np
import os
from collections import defaultdict
import random
from time import time

class LoadData(object):
    '''
    given the path of data, return the data format for FM
    Note that our data is stored as : 'label(0/1)' + ' feature1_num:feature1_value' + ' feature2_num:feature2_value' +... + ' feature_n_num:feature_n_value'

    Train_data: a dictionary,
    'Y' refers to a list of y values;
    'X' refers to a list matrix of row_length_features_M , X[i,j] == feature_id of sample i in column j
    'C' refers to a list matrix of row_length_features_M , C[i,j] == feature_value of sample i in column j
    Test_data: same as Train_data
    Validation_data: same as Train_data
    '''

    # Three files are needed in the path
    def __init__(self, path, dataset, loss_type):
        t1 = time()
        self.neg_dict = defaultdict(list)
        self.geo_list = list()
        self.path = path + dataset + "/"
        self.trainfile = self.path + dataset + "_train.txt"
        self.testfile = self.path + dataset + "_test.txt"
        self.validationfile = self.path + dataset + "_validation.txt"
        self.features_M = self.map_features()
        #self.features_M = self.get_feature_length(self.trainfile)
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )

        for key in self.neg_dict.keys():
            self.neg_dict[key] = list( set(self.geo_list) - set(self.neg_dict[key]) )
        print("\t ---- [%.4f]" % (time()-t1))

    def map_features(self): # map the feature entries in all files, kept in self.features dictionary
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        print("\tfeatures_M:", len(self.features))
        return  len(self.features)

    def read_features(self, file):  # read a feature file
        f = open(file)
        line = f.readline()
        i = len(self.features)
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                item = item.split(":")[0]
                if item not in self.features:
                    self.features[item] = i
                    i += 1

            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        X_, C_, Y_, Y_for_logloss = self.read_data(self.trainfile)
        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, C_, Y_for_logloss)
        else:
            Train_data = self.construct_dataset(X_, C_, Y_)
        print("# of training:", len(Y_))

        X_, C_, Y_, Y_for_logloss = self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, C_, Y_for_logloss)
        else:
            Validation_data = self.construct_dataset(X_, C_, Y_)
        print("# of validation:", len(Y_))

        X_, C_, Y_, Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, C_, Y_for_logloss)
        else:
            Test_data = self.construct_dataset(X_, C_, Y_)
        print("# of test:", len(Y_))

        return Train_data, Validation_data, Test_data

    def read_data(self, file):
        # read a data file. For a row, the first column goes into Y_;
        # the other columns become a row in X_ and entries are maped to indexs in self.features
        # data are stored as
        #    "y x1 x2 .... xn"
        X_ = []
        C_ = []
        Y_ = []
        Y_for_logloss = []
        with open(file) as f:
            line = f.readline()
            while line:
                items = line.strip().split(' ')
                Y_.append( 1.0*float(items[0]) )

                if float(items[0]) > 0:# > 0 as 1; others as 0
                    v = 1.0
                else:
                    v = 0.0
                Y_for_logloss.append( v )

                X_.append([self.features[item.split(":")[0]] for item in items[1:]])
                C_.append([float(item.split(":")[1]) for item in items[1:]])

                '''
                X_.append(list(range(0, len(items[1:]))))
                C_.append([item for item in items[1:]])
                '''
                k = len(items)
                up = k - 2  # user_id place
                cp = k - 1  # city_id place
                user_id = items[up].split(':')[0]
                geo_id = items[cp].split(':')[0]
                self.geo_list.append(self.features[geo_id])
                self.neg_dict[self.features[user_id]].append(self.features[geo_id])

                line = f.readline()

        return X_, C_, Y_, Y_for_logloss

    def construct_dataset(self, X_, C_, Y_):
        Data_Dic = {}
        X_lens = [len(line) for line in X_]
        indexs = np.argsort(X_lens)
        Data_Dic['Y'] = [Y_[i] for i in indexs]
        Data_Dic['C'] = [C_[i] for i in indexs]
        Data_Dic['X'] = [X_[i] for i in indexs]
        return Data_Dic

    '''
    def get_feature_length(self, file):
        with open(file) as f:
            line = f.readline()
            length = len(line.strip().split(' ')[1:])
        return length
    '''

    def neg_sampling(self, train, n = 2):
        length = len(train['X'])
        for i in range(0, length):
            for k in range(0,n):
                user = train['X'][i][-2]
                #print(user)
                while True:
                    neg_city = random.choice(self.neg_dict[user])
                    neg_city_ind = self.geo_list.index(neg_city)
                    if neg_city_ind < length: break
                x = train['X'][i][0:128] + train['X'][neg_city_ind][128:256]
                c = train['C'][i][0:128] + train['C'][neg_city_ind][128:256]
                y = 0

                train['X'].append(x)
                train['C'].append(c)
                train['Y'].append(y)

        return train

