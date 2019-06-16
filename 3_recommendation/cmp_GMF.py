'''
Created on Aug 9, 2016
Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  
@author: Xiangnan He (xiangnanhe@gmail.com)

@Modified by: Zhang Jiwen

'''
import numpy as np
import keras
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, Multiply, Reshape, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
import LoadData as DATA
from time import time, localtime, strftime
import multiprocessing as mp
import sys
import os
import logging
import math
import random
import argparse
from sklearn.metrics import mean_squared_error

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='cmp_test',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.05,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    parser.add_argument('--log_on', type=str, default='[\'lr\', \'batch_size\']',
                        help='the folder of logging')
    parser.add_argument('--log_path', type=str, default='plus-1-'+strftime("%m-%d", localtime()) ,
                        help='the folder of logging')
    return parser.parse_args()


#################### GET MODEL ####################
def get_model():
    # Input variables
    user_input = Input(shape=(128,), dtype='float32', name = 'user_input')
    item_input = Input(shape=(128,), dtype='float32', name = 'item_input')
    
    # Element-wise product of user and item embeddings
    predict_vector = Multiply()([user_input, item_input])
    
    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    # prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name = 'prediction')(predict_vector)
    prediction = Dense(1, activation="sigmoid", name="prediction", kernel_initializer="lecun_uniform")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], outputs=prediction)

    return model


#################### TRAIN MODEL ####################
def train_model(train, test, epochs, batch_size, verbose, model_out_file):
    # Init performance
    t1 = time()
    (rmse, ndcg) = evaluate_model(model, train)
    print('Init: RMSE = %.4f NDCG = %.4f [%.1f s]' % ( rmse, ndcg, time()-t1))
    logging.info('Init: RMSE = %.4f NDCG = %.4f [%.1f s]' % ( rmse, ndcg, time()-t1))
    
    # Train model
    best_rmse, best_iter = rmse, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, 1)
        
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)], #input
                         np.array(labels), # labels 
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        
        # Evaluation
        if epoch %verbose == 0:
            '''
            (rmse, ndcg) = evaluate_model(model, train)
            loss = hist.history['loss'][0]
            print('\tIteration %d [%.1f s]: --TrainSet RMSE = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, rmse, ndcg, loss, time() - t2))
            logging.info('\tIteration %d [%.1f s]: --TrainSet RMSE = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                         % (epoch, t2 - t1, rmse, ndcg, loss, time() - t2))
            '''
            # (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
            # hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            (rmse, ndcg) = evaluate_model(model, test)
            (rmse_train, ndcg_train) = evaluate_model(model, train)
            loss = hist.history['loss'][0]
            print('Iteration %d [%.1f s]: --TestSet RMSE = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]  --TrainSet RMSE = %.4f, NDCG = %.4f'
                  % (epoch,  t2-t1, rmse, ndcg, loss, time()-t2, rmse_train, ndcg_train))
            logging.info('Iteration %d [%.1f s]: --testSet RMSE = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]  --TrainSet RMSE = %.4f, NDCG = %.4f'
                         % (epoch,  t2-t1, rmse, ndcg, loss, time()-t2, rmse_train, ndcg_train))
            if rmse < best_rmse:
                best_rmse, best_iter = rmse, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  RMSE = %.4f. " %(best_iter, best_rmse))
    logging.info("End. Best Iteration %d:  RMSE = %.4f. " %(best_iter, best_rmse))
    if args.out > 0:
        print("The best GMF model is saved to %s" %(model_out_file))
        logging.info("The best GMF model is saved to %s" %(model_out_file))


def get_train_instances(train, neg_num):
    # add negative samples to Train Set
    # and return np.ndarray that satisfies the input constraints
    index_list = train['X']
    value_list = train['C']
    users = []
    items = []
    labels = []

    # prepare for neg_sampling
    geo_list = [index_list[i][-1] for i in range(0, len(index_list))]
    geo_set = set(geo_list)
    user_geo_dict = {}
    if neg_num :
        for i in range(0, len(index_list)):
            id = index_list[i][-2]
            if id in user_geo_dict:
                user_geo_dict[id].add(index_list[i][-1])
            else:
                user_geo_dict[id] = {index_list[i][-1]}

        for key in user_geo_dict.keys():
            user_geo_dict[key] = list(geo_set - user_geo_dict[key])

    # ns = list(range(0, len(index_list)))
    for i in range(0, len(index_list)):
        user_num = index_list[i][-2]
        val = value_list[i]

        users.append(val[0:128])
        items.append(val[128:256])
        labels.append(train['Y'][i])
        if train['Y'][i] == 0: continue

        ### negative sampling ######
        for j in range(0, neg_num):
            # random.shuffle(ns)
            neg_geo = random.choice(user_geo_dict[user_num])
            neg_i = geo_list.index(neg_geo)
            neg_val = value_list[neg_i]
            users.append(val[0:128])
            items.append(neg_val[128:256])
            labels.append(0)

    return users, items, labels


#################### EVALUATE MODEL ####################
def evaluate_model(model, test):
    # predictions = model.predict([users, items, batch_size=100, verbose=0)
    users, items, labels = get_train_instances(test, 0)
    ids = [indx[257] for indx in test['X']]
    predictions = model.predict([np.array(users), np.array(items)], batch_size = 100, verbose = 0)

    y_pred = np.reshape(predictions, (len(labels),))
    y_true = labels

    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    ndcg = NDCG().ndcg(ids, y_true, y_pred, k=10)

    return rmse, ndcg


class NDCG():
    def ndcg(self, users, y_true, y_score, k=5):
        scores = []
        y_true_dict, y_score_dict = self.user_detect(users, y_true, y_score)

        # Iterate over each y_value_true and compute the DCG score
        for key in y_true_dict.keys():
            y_value_true, y_value_score = y_true_dict[key], y_score_dict[key]
            actual = self.dcg_score(y_value_true, y_value_score, k)
            best = self.dcg_score(y_value_true, y_value_true, k)
            if best:
                scores.append(actual / best)
        return np.mean(scores)

    def user_detect(self, users, y_true, y_score):
        y_true_dict, y_score_dict = {}, {}
        for (user, y_t, y_s) in zip(users, y_true, y_score):
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


if __name__ == '__main__':
    args = parse_args()
    learner = args.learner
    learning_rate = args.lr
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose

    args_str = str(args)
    model_out_file = 'Pretrain/%s_GMF_%d.h5' % (args.dataset, time())
    log_name = "".join([args_str[args_str.find(string + "="):].split(",")[0] for string in eval(args.log_on)])
    #print(log_name)
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(message)s',
                        datefmt='%a %d %b %Y %H:%M %S',
                        filename=os.path.join(args.log_path, "GMF%s.txt" % log_name))
    print(args)
    logging.info("==============")
    logging.info(args)

    # Loading data
    t1 = time()
    data = DATA.LoadData(args.path, args.dataset, 'binary_crossentropy')
    train = data.Train_data
    test = data.Test_data
    print("Load data done [%.1f s]. #train=%d " %( time()-t1, len(train['C']) ) )
    logging.info("Load data done [%.1f s]. #train=%d " %( time()-t1, len(train['C']) ))
    
    # Build model
    model = get_model()
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    #print(model.summary())

    train_model(train, test, epochs, batch_size, verbose, model_out_file)
