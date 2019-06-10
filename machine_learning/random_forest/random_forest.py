# -*- coding: utf-8 -*-
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
from dataset_split import DataSplit

test_m = 5000
test_n = 100000
X_train, _, y_train, _ = DataSplit().split_dataset(
    test_malicious=test_m, test_normal=test_n)
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.1, random_state=1024)
"""
train_size = len(y_train)
train_m = sum(y_train)
train_n = train_size - train_m
print(">>> Train set: M=%d, N=%d" % (train_m, train_n))
"""
test_size = len(y_test)
test_m = sum(y_test)
test_n = test_size - test_m
print(">>> Test set: M=%d, N=%d" % (test_m, test_n))
"""

model = RandomForestClassifier(
    n_estimators=100, max_depth=30, random_state=1024)

# train and evaluate
t0 = time()
model.fit(X_train, y_train)
t1 = time()
print(">>> training time: ", t1 - t0)

dump(model, 'RandomForestClassifier.joblib')
