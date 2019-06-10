# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys
sys.path.append("..")
from include.Dataset import Dataset


class DataSplit():
    def __init__(self):
        # load dataset
        self.X, self.y = Dataset().get_dataset()

    def split_dataset(self,
                      test_normal=250000,
                      test_malicious=5000,
                      random_state=1024):
        M = sum(self.y)  # number of malicious samples
        N = len(self.y) - M  # number of normal samples
        print(">>> normal=%d, malicious=%d" % (N, M))
        malicious_y = self.y[self.y.values == 1]
        malicious_X = self.X.loc[malicious_y.index.to_list()]
        malicious_X_train, malicious_X_test, malicious_y_train, malicious_y_test = train_test_split(
            malicious_X,
            malicious_y,
            test_size=test_malicious,
            random_state=random_state)

        normal_y = self.y[self.y.values == 0]
        normal_X = self.X.loc[normal_y.index.to_list()]
        normal_X_train, normal_X_test, normal_y_train, normal_y_test = train_test_split(
            normal_X,
            normal_y,
            test_size=test_normal,
            random_state=random_state)

        X_train = np.concatenate((normal_X_train, malicious_X_train))
        X_test = np.concatenate((normal_X_test, malicious_X_test))
        y_train = np.concatenate((normal_y_train, malicious_y_train))
        y_test = np.concatenate((normal_y_test, malicious_y_test))
        return X_train, X_test, y_train, y_test