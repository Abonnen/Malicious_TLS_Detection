# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
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
        param_grid = {'max_depth': range(10, 31, 5)}
        clf = GridSearchCV(
            estimator=RandomForestClassifier(
                n_estimators=100,
                random_state=2019,
                oob_score=True,
                warm_start=True),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.X, self.y)

        print(clf.best_params_)
        print(clf.best_score_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('n_estimators.csv', 'w+') as f:
            cv_result.to_csv(f)

        return clf.best_params_

    def gsearch_2(self):
        param_grid = {'max_depth': range(5, 11)}
        clf = GridSearchCV(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=2019),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.X, self.y)

        print(">>> max_depth:")
        print(clf.best_params_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('max_depth.csv', 'w+') as f:
            cv_result.to_csv(f)

        return clf.best_params_

    def gsearch_3(self):
        param_grid = {'max_features': range(5, 11)}
        clf = GridSearchCV(
            estimator=RandomForestClassifier(
                n_estimators=100, max_depth=30, random_state=2019),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.X, self.y)

        print(">>> max_features:")
        print(clf.best_params_)
        print(clf.best_score_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('max_features.csv', 'w+') as f:
            cv_result.to_csv(f)

        return clf.best_params_

    def gsearch_4(self):
        param_grid = {
            'min_samples_leaf': range(5, 21, 5),
            'min_samples_split': range(5, 126, 20)
        }
        clf = GridSearchCV(
            estimator=RandomForestClassifier(
                n_estimators=100, max_depth=30, random_state=2019, n_jobs=4),
            param_grid=param_grid,
            scoring='f1',
            return_train_score=True,
            cv=10)
        clf.fit(self.eval_X, self.eval_y)

        print(clf.best_params_)
        print(clf.best_score_)
        cv_result = pd.DataFrame.from_dict(clf.cv_results_)
        with open('min_samples.csv', 'w+') as f:
            cv_result.to_csv(f)

        return clf.best_params_


gs = GridSeach()
gs.gsearch_4()