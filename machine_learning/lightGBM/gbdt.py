# -*- coding: utf-8 -*-
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
from time import time
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from joblib import dump
from dataset_split import DataSplit

import sys
sys.path.append("..")
from include.AlarmMetric import false_positive_rate, false_negative_rate, false_discovery_rate

test_m = 5000
test_n = 100000
X_train, _, y_train, _ = DataSplit().split_dataset(
    test_malicious=test_m, test_normal=test_n)

X_train, X_eval, y_train, y_eval = train_test_split(
    X_train, y_train, test_size=0.1, random_state=2019)
"""
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.1, random_state=2019)
"""
train_size = len(y_train)
train_m = sum(y_train)
train_n = train_size - train_m
print(">>> Train set: M=%d, N=%d" % (train_m, train_n))

eval_size = len(y_eval)
eval_m = sum(y_eval)
eval_n = eval_size - eval_m
print(">>> Validation set: M=%d, N=%d" % (eval_m, eval_n))
"""
test_size = len(y_test)
test_m = sum(y_test)
test_n = test_size - test_m
print(">>> Test set: M=%d, N=%d" % (test_m, test_n))
"""

model = LGBMClassifier(
    boosting_type='gbdt',
    importance_type='split',
    num_leaves=50,
    min_child_samples=100,
    max_depth=8,
    random_state=2019)

# train and evaluate
t0 = time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_eval, y_eval)],
    eval_names='eval',
    eval_metric=['logloss', 'auc'],
    early_stopping_rounds=100)
t1 = time()
print(">>> training time: ", t1 - t0)

dump(model, 'LGBMClassifier.joblib')
"""
for i in range(10):
    graph = lgb.create_tree_digraph(model, tree_index=i, name=str(i))
    graph.render(view=False)
"""
