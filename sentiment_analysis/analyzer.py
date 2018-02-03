# coding = utf-8
import jieba as jb
import requests
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import tensorflow as tf
import torch
import copy

def collect_dataset():
    response = requests.get('https://https://github.com/jswhy/general_ml' +
                            'movie_comment/training_data/' +
                            'comment1.csv')
    lines = response.text.splitlines()
    data = []
    for item in lines:
        item = item.split(',')
        # item = item.split(' ')
        # item = item.split('；')
        # item = item.split('/// ')
        data.append(item)
    data.pop(0)  # This is for removing the labels from the list
    dataset = np.matrix(data)
    return dataset


class Hoffman_Tree:
    '''
    build a huffman tree to store words
    '''

    def __init__(self, depth=5, min_leaf_size=5):
        self.depth = depth
        self.decision_boundary = 0
        self.left = None
        self.right = None
        self.min_leaf_size = min_leaf_size
        self.prediction = None

    def mean_squared_error(self, labels, prediction):
        '''
           loss function
        '''
        if labels.ndim != 1:
            print("Error: Input labels must be one dimensional")

        return np.mean((labels - prediction) ** 2)

    def train(self, X, y):
        '''
           build a huffman tree to store words
           '''
        if X.ndim != 1:
            print("Error: Input data set must be one dimensional")
            return
        if len(X) != len(y):
            print("Error: X and y have different lengths")
            return
        if y.ndim != 1:
            print("Error: Data set labels must be one dimensional")
            return

        if len(X) < 2 * self.min_leaf_size:
            self.prediction = np.mean(y)
            return

        if self.depth == 1:
            self.prediction = np.mean(y)
            return

        best_split = 0
        min_error = self.mean_squared_error(X, np.mean(y)) * 2

        for i in range(len(X)):
            if len(X[:i]) < self.min_leaf_size:
                continue
            elif len(X[i:]) < self.min_leaf_size:
                continue
            else:
                error_left = self.mean_squared_error(X[:i], np.mean(y[:i]))
                error_right = self.mean_squared_error(X[i:], np.mean(y[i:]))
                error = error_left + error_right
                if error < min_error:
                    best_split = i
                    min_error = error

        if best_split != 0:
            left_X = X[:best_split]
            left_y = y[:best_split]
            right_X = X[best_split:]
            right_y = y[best_split:]

            self.decision_boundary = X[best_split]
            self.left = Hoffman_Tree(depth=self.depth - 1, min_leaf_size=self.min_leaf_size)
            self.right = Hoffman_Tree(depth=self.depth - 1, min_leaf_size=self.min_leaf_size)
            self.left.train(left_X, left_y)
            self.right.train(right_X, right_y)
        else:
            self.prediction = np.mean(y)

        return

    def predict(self, x):

        if self.prediction is not None:
            return self.prediction
        elif self.left or self.right is not None:
            if x >= self.decision_boundary:
                return self.right.predict(x)
            else:
                return self.left.predict(x)
        else:
            print("Error: Decision tree not yet trained")
            return None

    def cal_pyx(self, X, y):
        result = 0.0
        for x in X:
            if self.fxy(x, y):
                id = self.xy2id[(x, y)]
                result += self.w[id]
        return (math.exp(result), y)

    def cal_probality(self, X):

        Pyxs = [(self.cal_pyx(X, y)) for y in self.Y_]
        Z = sum([prob for prob, y in Pyxs])
        return [(prob / Z, y) for prob, y in Pyxs]

    def cal_EPx(self):
        '''
        desired value
        '''

        for i, X in enumerate(self.X_):
            Pyxs = self.cal_probality(X)

            for x in X:
                for Pyx, y in Pyxs:
                    if self.fxy(x, y):
                        id = self.xy2id[(x, y)]

                        self.EPx[id] += Pyx * (1.0 / self.N)

    def fxy(self, x, y):
        return (x, y) in self.xy2id

    def rebuild_features(features):
        '''
        feather transform
         '''
        new_features = []
        for feature in features:
            new_feature = []
            for i, f in enumerate(feature):
                new_feature.append(str(i) + '_' + str(f))
            new_features.append(new_feature)
            return new_features


def run_steep_gradient_descent(data_x, data_y,
                               len_data, alpha, theta):
    n = len_data

    prod = np.dot(theta, data_x.transpose())
    prod -= data_y.transpose()
    sum_grad = np.dot(prod, data_x)
    theta = theta - (alpha / n) * sum_grad
    return theta


def sum_of_square_error(data_x, data_y, len_data, theta):
    error = 0.0
    prod = np.dot(theta, data_x.transpose())
    prod -= data_y.transpose()
    sum_elem = np.sum(np.square(prod))
    error = sum_elem / (2 * len_data)
    return error


def run_linear_regression(data_x, data_y):
    iterations = 100000
    alpha = 0.0001550

    no_features = data_x.shape[1]
    len_data = data_x.shape[0] - 1

    theta = np.zeros((1, no_features))

    for i in range(0, iterations):
        theta = run_steep_gradient_descent(data_x, data_y,
                                           len_data, alpha, theta)
        error = sum_of_square_error(data_x, data_y, len_data, theta)
        print('At Iteration %d - Error is %.5f ' % (i + 1, error))

    return theta


def main():
    data = collect_dataset()

    len_data = data.shape[0]
    data_x = np.c_[np.ones(len_data), data[:, :-1]].astype(float)
    data_y = data[:, -1].astype(float)

    theta = run_linear_regression(data_x, data_y)
    len_result = theta.shape[1]
    print('Resultant Feature vector : ')
    for i in range(0, len_result):
        print('%.5f' % (theta[0, i]))


str = input("please enter your input comment:")
seg_list = jb.cut(str, cut_all=True)
print("All word analyzer:", "/ ".join(seg_list))
print("================================")
seg_list = jb.cut(str, cut_all=False)
print("The best analyzer:", "/ ".join(seg_list))
