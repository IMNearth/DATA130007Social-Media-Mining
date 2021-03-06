'''
Tensorflow implementation of Factorization Machines (FM) as described in:
Xiangnan He, Tat-Seng Chua. Neural Factorization Machines for Sparse Predictive Analytics. In Proc. of SIGIR 2017.

Note that the original paper of FM is: Steffen Rendle. Factorization Machines. In Proc. of ICDM 2010.

@author:
Xiangnan He (xiangnanhe@gmail.com)
Lizi Liao (liaolizi.llz@gmail.com)

@Modified by: Zhang Jiwen at 2019/05/22

@references:https://github.com/hexiangnan/neural_factorization_machine
@Inspired by: Zhankui He
'''

import math
import os
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from time import time, strftime, localtime
import argparse
import LoadData as DATA
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from multiprocessing import Pool, cpu_count
import logging
# import gc

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

####### FM #######

def _padding(data):
    # for multiprocessing in FM.batch_gen()
    X, C, Y = data['X'], data['C'], data['Y']
    max_length = max([len(x) for x in X])
    X_ = [ x + [0 for _ in range(max_length - len(x))] for x in X]
    C_ = [ c + [0 for _ in range(max_length - len(c))] for c in C]
    Y_ = [[y] for y in Y]
    return {'X': X_, 'C':C_, 'Y': Y_ }


class FM(BaseEstimator, TransformerMixin):
    def __init__(self, features_M, pretrain_flag, save_file, hidden_factor,
                 loss_type, epoch, batch_size, learning_rate, lamda_bilinear, keep,
                 optimizer_type, batch_norm, verbose, dataset, args, random_seed=2016):
        # bind params to class
        self.args = args
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_factor = hidden_factor
        self.save_file = save_file
        self.pretrain_flag = pretrain_flag
        self.loss_type = loss_type
        self.features_M = features_M
        self.lamda_bilinear = lamda_bilinear
        self.keep = keep
        self.epoch = epoch
        self.random_seed = random_seed
        self.optimizer_type = optimizer_type
        self.batch_norm = batch_norm
        self.verbose = verbose
        # performance of each epoch
        self.train_rmse, self.valid_rmse, self.test_rmse = [], [], []
        self.dataset = dataset

        # init all variables in a tensorflow graph
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.train_features = tf.placeholder(tf.int32, shape=[None, None])  # None * features_M
            self.train_coeff = tf.placeholder(tf.float32, shape=[None, None])  # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape=[None, 1])  # None * 1
            self.dropout_keep = tf.placeholder(tf.float32)
            self.train_phase = tf.placeholder(tf.bool)

            # Variables.
            self.weights = self._initialize_weights()

            # Model.
            # _________ sum_square part _____________
            # get the summed up embeddings of features.
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            self.train_coeff_expend = tf.expand_dims(self.train_coeff, -1)  # None * feature_M * 1
            self.coeff_embeddings = nonzero_embeddings * self.train_coeff_expend
            self.summed_features_emb = tf.reduce_sum(self.coeff_embeddings, 1)  # None * K
            # get the element-multiplication
            self.summed_features_emb_square = tf.square(self.summed_features_emb)  # None * K

            # _________ square_sum part _____________
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1)  # None * K

            # ________ FM __________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.batch_norm:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep)  # dropout at the FM layer

            # _________out _________
            Bilinear = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features),1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Bilinear, self.Feature_bias, Bias])  # None * 1

            # Compute the loss.
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(
                        tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out)
                if self.lamda_bilinear > 0:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
                                                           scope=None) + tf.contrib.layers.l2_regularizer(
                        self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.contrib.losses.log_loss(self.out, self.train_labels, weight=1.0, epsilon=1e-07,
                                                           scope=None)

            # Optimizer.
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                        epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                           initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(
                    self.loss)

            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

            ''' if in GPU
            # init
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.3
            self.sess = tf.Session(config=config)
            self.sess.run(init)
            '''

            # number of params
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()  # shape is an array of tf.Dimension
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def _initialize_weights(self):
        all_weights = dict()
        if self.pretrain_flag > 0:
            weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
            pretrain_graph = tf.get_default_graph()
            feature_embeddings = pretrain_graph.get_tensor_by_name('feature_embeddings:0')
            feature_bias = pretrain_graph.get_tensor_by_name('feature_bias:0')
            bias = pretrain_graph.get_tensor_by_name('bias:0')
            with tf.Session() as sess:
                weight_saver.restore(sess, self.save_file)
                fe, fb, b = sess.run([feature_embeddings, feature_bias, bias])
            all_weights['feature_embeddings'] = tf.Variable(fe, dtype=tf.float32)
            all_weights['feature_bias'] = tf.Variable(fb, dtype=tf.float32)
            all_weights['bias'] = tf.Variable(b, dtype=tf.float32)
        else:
            all_weights['feature_embeddings'] = tf.Variable(
                tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01),
                name='feature_embeddings')  # features_M * K
            all_weights['feature_bias'] = tf.Variable(
                tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
            all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        return all_weights

    def batch_norm_layer(self, x, train_phase, scope_bn):
        # Note: the decay parameter is tunable
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z

    def partial_fit(self, data):  # fit a batch
        feed_dict = {self.train_features: data['X'],
                     self.train_labels: data['Y'],
                     self.train_coeff: data['C'],
                     self.dropout_keep: self.keep,
                     self.train_phase: True}
        # 将 ‘ feed_dict ’ 中的值替换为相应的输入值
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    '''
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep,
                     self.train_phase: True}

        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
    '''

    def get_random_block_from_data(self, data, batch_size):  # generate a random block of training data
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        # forward get sample
        i = start_index
        t1 = time()

        Y = [[data['Y'][i + j]] for j in range(batch_size)]
        X = [data['X'][i + j] for j in range(batch_size)]
        C = [data['C'][i + j] for j in range(batch_size)]

        t2 = time()
        #print("time1:", t2-t1)

        X = self.padding(X)
        C = self.padding(C)

        # print("time2:", time() - t2)
        return {'X': X, 'Y': Y, 'C': C}
    '''
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X, Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}
    '''
    
    '''
    def batch_gen(self, data, batch_size, total_batch):
        batches = [{'X': data['X'][j*batch_size: (j+1)*batch_size],
                    'Y': data['Y'][j*batch_size: (j+1)*batch_size],
                    'C': data['X'][j*batch_size: (j+1)*batch_size]}
                    for j in range(total_batch)]

        pool = Pool(cpu_count())
        res = pool.map(_padding, batches)
        pool.close()
        pool.join()

        return res
    '''

    def padding(self, data, pad=0):
        max_length = max([len(d) for d in data])
        data = [d+[pad for _ in range(max_length - len(d))] for d in data]
        return data

    def shuffle_in_unison_scary(self, a, b, c):  # shuffle three lists simutaneously
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def train(self, Train_data___, Validation_data, Test_data):  # fit a dataset
        # Check Init performance
        Train_data = self.dataset.neg_sampling(Train_data___,  n=self.args.neg_num)

        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train= %s, validation= %s, test= %s [%.1f s]" % (init_train, init_valid, init_test, time() - t2))
            logging.info("Init: \t train: %s, validation: %s, test: %s [%.1f s]" % (init_train, init_valid, init_test, time() - t2))

        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y'], Train_data['C'])
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            for i in range(total_batch):
                # generate a batch
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # Fit training
                self.partial_fit(batch_xs)
            t2 = time()

            # output validation
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)

            self.train_rmse.append(train_result)
            self.valid_rmse.append(valid_result)
            self.test_rmse.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [%.1f s]\ttrain: %s, validation: %s, test: %s [%.1f s]"
                      % (epoch + 1, t2 - t1, train_result, valid_result, test_result, time() - t2))
                logging.info("Epoch %d [%.1f s]\ttrain: %s, validation: %s, test: %s [%.1f s]"
                             % (epoch + 1, t2 - t1, train_result, valid_result, test_result, time() - t2))
            if self.eva_termination(self.valid_rmse):
                break

        if self.pretrain_flag < 0:
            print("Save model to file as pretrain.")
            logging.info("Save model to file as pretrain. --> %s" % self.save_file)
            self.saver.save(self.sess, self.save_file)

    def eva_termination(self, valid):
        if self.loss_type == 'square_loss':
            if len(valid) > 5:
                if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                    return True
        else:
            if len(valid) > 5:
                if valid[-1] < valid[-2] and valid[-2] < valid[-3] and valid[-3] < valid[-4] and valid[-4] < valid[-5]:
                    return True
        return False

    def evaluate(self, data):  # evaluate the results for an input set
        num_example = len(data['Y'])
        total_batch = int(np.ceil(len(data['Y']) / self.batch_size))
        predictions = []
        for b in range(total_batch):
            begin, end = b*self.batch_size, (b+1)*self.batch_size
            feed_dict = {self.train_features: self.padding(data['X'][begin: end]),
                         self.train_coeff: self.padding(data['C'][begin: end]),
                         self.train_labels: [[y] for y in data['Y'][begin: end]],
                         self.dropout_keep: 1.0,
                         self.train_phase: False}
            predictions.append(self.sess.run((self.out), feed_dict=feed_dict))
        predictions = np.concatenate(predictions)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        if self.loss_type == 'square_loss':
            y_pred = np.clip(y_pred, a_min=min(y_true), a_max=max(y_true))
            RMSE = math.sqrt(mean_squared_error(y_true, y_pred))
            NDCG = self.ndcg_score(data['X'], y_true, y_pred, k=10)
            return "RMSE %.4f, NDCG %.4f" % (RMSE, NDCG)
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred) # I haven't checked the log_loss
            return logloss

    # prepare for ndcg computation
    def user_detect(self, features, y_true, y_score):
        y_true_dict, y_score_dict = {}, {}
        for (f, y_t, y_s) in zip(features, y_true, y_score):
            user = f[-2]
            if user in y_true_dict:
                y_true_dict[user].append(y_t)
                y_score_dict[user].append(y_s)
            else:
                y_true_dict[user] = [y_t]
                y_score_dict[user] = [y_s]

        return y_true_dict, y_score_dict

    def dcg_score(self, y_true, y_score, k=5):
        order = np.argsort(y_score)[::-1]
        y_true = np.take(y_true, order[:k])
        gain = 2 ** y_true - 1
        discounts = np.log2(np.arange(len(y_true)) + 2)
        return np.sum(gain / discounts)

    def ndcg_score(self, features, y_true, y_score, k=5):
        scores = []
        y_true_dict, y_score_dict = self.user_detect(features, y_true, y_score)

        # Iterate over each y_value_true and compute the DCG score
        for key in y_true_dict.keys():
            y_value_true, y_value_score = y_true_dict[key], y_score_dict[key]
            actual = self.dcg_score(y_value_true, y_value_score, k)
            best = self.dcg_score(y_value_true, y_value_true, k)
            if best:
                scores.append(actual / best)
        return np.mean(scores)


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run FM.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='test3',
                        help='Choose a dataset.')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--pretrain', type=int, default=-1,
                        help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors.')
    parser.add_argument('--lamda', type=float, default=0,
                        help='Regularizer for bilinear part.')
    parser.add_argument('--keep_prob', type=float, default=0.5,
                        help='Keep probility (1-dropout_ratio) for the Bi-Interaction layer. 1: no dropout')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--loss_type', nargs='?', default='square_loss',
                        help='Specify a loss type (square_loss or log_loss).')
    parser.add_argument('--optimizer', nargs='?', default='AdagradOptimizer',
                        help='Specify an optimizer type (AdamOptimizer, AdagradOptimizer, GradientDescentOptimizer, MomentumOptimizer).')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show the results per X epochs (0, 1 ... any positive integer)')
    parser.add_argument('--batch_norm', type=int, default=0,
                        help='Whether to perform batch normaization (0 or 1)')
    parser.add_argument('--neg_num', type=int, default=2,
                        help='Number of negative samples for each user in train data')
    parser.add_argument('--log_path', type=str, default='plus-1-'+strftime("%m-%d", localtime()),
                        help='the folder of logging')
    parser.add_argument('--log_on', type=str, default='[\'lr\', \'batch_size\']',
                        help='the folder of logging')

    return parser.parse_args()


if __name__ == '__main__':
    # Data loading
    args = parse_args()

    args_str = str(args)
    log_name = "".join([args_str[args_str.find(string + "="):].split(",")[0] for string in eval(args.log_on)])
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M %S',
                        filename=os.path.join(args.log_path, "FM%s.txt" % log_name))
    print(args)
    logging.info("==============")
    logging.info(args)

    data = DATA.LoadData(args.path, args.dataset, args.loss_type)
    if args.verbose > 0:
        print(
            "FM: dataset=%s, factors=%d, loss_type=%s, #epoch=%d, batch=%d, lr=%.4f, lambda=%.1e, keep=%.2f, optimizer=%s, batch_norm=%d"
            % (args.dataset, args.hidden_factor, args.loss_type, args.epoch, args.batch_size, args.lr, args.lamda,
               args.keep_prob, args.optimizer, args.batch_norm))

    save_file = '../pretrain/0609_FM_%s_h%d_lr%.4f_b%d' % (args.dataset, args.hidden_factor, args.lr, args.batch_size)

    # Training
    t1 = time()
    model = FM(data.features_M, args.pretrain, save_file, args.hidden_factor, args.loss_type, args.epoch,
               args.batch_size, args.lr, args.lamda, args.keep_prob, args.optimizer, args.batch_norm,
               args.verbose, data, args)
    model.train(data.Train_data, data.Validation_data, data.Test_data)

    # Find the best validation result across iterations
    best_valid_score = 0
    if args.loss_type == 'square_loss':
        best_valid_score = min(model.valid_rmse)
    elif args.loss_type == 'log_loss':
        best_valid_score = max(model.valid_rmse)
    best_epoch = model.valid_rmse.index(best_valid_score)
    print("Best Iter(validation)= %d\t train = %s, valid = %s, test = %s [%.1f s]"
          % (best_epoch + 1, model.train_rmse[best_epoch], model.valid_rmse[best_epoch], model.test_rmse[best_epoch],time() - t1))
