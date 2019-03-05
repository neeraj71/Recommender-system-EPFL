import numpy as np
import os
import pickle


def blend_train(als=None, sgd=None, user__mean=None, knn_user=None, knn_item=None, svd=None, svdpp=None, nnmf=None, alpha=None, label=None):
    """
    This function train blending model for our prediction of algorithms 
    :param als: is prediction array of als algrithm with shape of n*1 
    :param sgd: is prediction array of sgd algrithm with shape of n*1 
    :param user__mean: is prediction array of user__mean algrithm with shape of n*1 
    :param knn_user is prediction array of knn_user algrithm with shape of n*1 
    :param knn_item is prediction array of knn_item algrithm with shape of n*1 
    :param svd is prediction array of svd algrithm with shape of n*1 
    :param svdpp is prediction array of svd algrithm with shape of n*1 
    :param nnmf: is prediction array of nnmf algrithm with shape of n*1 
    :param alpha: the parameter for regualrize
    :param label: the true raking 
    :return: weight w and RMSE of blending
    """

    m_train = np.concatenate((als, sgd, user__mean, knn_user, knn_item, svd, svdpp, nnmf), axis=1)
    y_train = lable
    # Ridge Regression
    w = np.linalg.solve(m_train.T.dot(m_train) + alpha * np.eye(m_train.shape[1]), m_train.T.dot(y_train))
    
    y_predict_train = m_train.dot(w)
    # Cut predictions that are too high and too low
    for i in range(len(y_predict_train)):
        y_predict_train[i] = min(5, np.round(y_predict_train[i]))
        y_predict_train[i] = max(1, np.round(y_predict_train[i]))

    return w, np.sqrt(np.mean((y_train - y_predict_train) ** 2))
        # with open('../data/predictions/merge_prediction.csv', 'w') as output:
        #     output.write('Id,Prediction\n')
        #     for i, (row, col) in enumerate(nnz_test):
        #         output.write('r{}_c{},{}\n'.format(row + 1, col + 1, y_predict_test[i]))

def blend_pred(als, nnmf, w, lable=None, predict=False):

    m_train = np.concatenate((als, nnmf), axis=1)
    y_train = lable
    y_predict_train = m_train.dot(w)
    # Cut predictions that are too high and too low
    for i in range(len(y_predict_train)):
        y_predict_train[i] = min(5, np.round(y_predict_train[i]))
        y_predict_train[i] = max(1, np.round(y_predict_train[i]))

    return y_predict_train