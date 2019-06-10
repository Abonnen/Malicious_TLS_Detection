# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, train_test_split
import sys
sys.path.append("..")
from include.Dataset import Dataset


class GridSeach():
    def __init__(self):
        self.eval_X = None
        self.eval_y = None
        self.selected_feature = []
        self.X, self.y = Dataset().get_dataset()
        self.split_dataset()

    def split_dataset(self):
        _, self.eval_X, _, self.eval_y = train_test_split(
            self.X, self.y, test_size=0.2, random_state=2019)

    def gsearch_1(self):
        param_grid = {
            'min_child_samples': range(100, 106),
            'max_depth': range(6, 9)
        }
        clf = GridSearchCV(
            estimator=LGBMClassifier(
                boosting_type='gbdt',
                importance_type='split',
                num_leaves=50,
                random_state=2019),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.eval_X, self.eval_y)

        print(clf.best_params_)
        print(clf.best_score_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('min_child_samples_max_depth_2.csv', 'w+') as f:
            cv_result.to_csv(f)

    def gsearch_2(self):
        param_grid = {'min_child_samples': range(20, 201, 20)}
        clf = GridSearchCV(
            estimator=LGBMClassifier(num_leaves=70, random_state=2019),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.eval_X, self.eval_y)

        print(clf.best_params_)
        print(clf.best_score_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('min_child_samples.csv', 'w+') as f:
            cv_result.to_csv(f)

    def gsearch_3(self):
        param_grid = {'max_depth': range(6, 11)}
        clf = GridSearchCV(
            estimator=LGBMClassifier(
                num_leaves=70, min_child_samples=40, random_state=2019),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.eval_X, self.eval_y)

        print(clf.best_params_)
        print(clf.best_score_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('max_depth.csv', 'w+') as f:
            cv_result.to_csv(f)


gs = GridSeach()
gs.gsearch_1()