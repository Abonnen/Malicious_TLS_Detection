# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
from joblib import dump, load
from treeinterpreter import treeinterpreter as ti

import sys
sys.path.append("..")
from include.Dataset import Dataset


class FeatureSelection():
    def __init__(self):
        self.X = None
        self.y = None
        self.eval_X = None
        self.eval_y = None
        self.test_X = None
        self.test_y = None
        self.selected_feature = []
        ds = Dataset()
        self.X, self.y = ds.get_dataset()
        self.test_X, self.test_y = ds.get_testset()
        self.feature_column = ds.get_column_name()

    def interpret(self):
        feature_set = [
            'avg_cert_path', 'avg_cert_valid_day', 'avg_domain_name_length',
            'avg_duration', 'avg_IPs_in_DNS', 'avg_pkts', 'avg_size',
            'avg_time_diff', 'avg_TTL', 'avg_valid_cert_percent',
            'cert_key_type', 'cert_sig_alg', 'cipher_suite_server',
            'is_CNs_in_SNA_dns', 'is_O_in_issuer', 'is_O_in_subject',
            'is_SNIs_in_SNA_dns', 'is_ST_in_issuer', 'is_ST_in_subject',
            'max_duration', 'max_time_diff', 'number_of_domains_in_cert',
            'number_of_flows', 'packet_loss', 'percent_of_established_state',
            'percent_of_valid_cert', 'percent_of_std_duration',
            'recv_sent_pkts_ratio', 'recv_sent_size_ratio', 'resumed',
            'SNI_ssl_ratio', 'ssl_flow_ratio', 'ssl_version',
            'std_cert_valid_day', 'std_domain_name_length', 'std_time_diff',
            'subject_CN_is_IP', 'subject_is_com', 'subject_only_CN',
            'x509_ssl_ratio'
        ]
        common_feature = [
            'avg_cert_path', 'avg_cert_valid_day', 'avg_domain_name_length',
            'avg_duration', 'avg_IPs_in_DNS', 'avg_pkts', 'avg_size',
            'avg_time_diff', 'avg_TTL', 'avg_valid_cert_percent',
            'cert_key_type', 'cert_sig_alg', 'cipher_suite_server',
            'is_CNs_in_SNA_dns', 'is_O_in_issuer', 'is_O_in_subject',
            'is_ST_in_subject', 'max_duration', 'max_time_diff',
            'number_of_domains_in_cert', 'number_of_flows', 'packet_loss',
            'recv_sent_pkts_ratio', 'recv_sent_size_ratio', 'ssl_version',
            'std_domain_name_length', 'std_time_diff', 'subject_only_CN'
        ]

        plus_feature = [
            'avg_cert_path', 'avg_cert_valid_day', 'avg_domain_name_length',
            'avg_duration', 'avg_IPs_in_DNS', 'avg_pkts', 'avg_size',
            'avg_time_diff', 'avg_TTL', 'avg_valid_cert_percent',
            'cert_key_type', 'cert_sig_alg', 'cipher_suite_server',
            'is_CNs_in_SNA_dns', 'is_O_in_issuer', 'is_O_in_subject',
            'is_ST_in_subject', 'max_duration', 'max_time_diff',
            'number_of_domains_in_cert', 'number_of_flows', 'packet_loss',
            'recv_sent_pkts_ratio', 'recv_sent_size_ratio', 'ssl_version',
            'std_domain_name_length', 'std_time_diff', 'subject_only_CN',
            'resumed', 'SNI_ssl_ratio'
        ]
        # model = RandomForestClassifier(n_estimators=100, random_state=2019)
        rfc = RandomForestClassifier(n_estimators=100, random_state=2019)
        X, train_X, y, train_y = train_test_split(
            self.X, self.y, test_size=200000, random_state=1024)
        _, eval_X, _, eval_y = train_test_split(
            X, y, test_size=200000, random_state=1024)

        train_X = train_X[feature_set].reset_index(drop=True)
        eval_X = eval_X[feature_set].reset_index(drop=True)

        train_X_new = train_X[plus_feature].reset_index(drop=True)
        eval_X_new = eval_X[plus_feature].reset_index(drop=True)
        eval_y = eval_y.reset_index(drop=True)

        # model.fit(train_X, train_y)
        # dump(model, 'rf_model_union.joblib')
        rfc.fit(train_X_new, train_y)
        dump(rfc, 'rf_model_plus.joblib')
        model = load('rf_model_union.joblib')
        # rfc = load('rf_model_plus.joblib')
        pred = model.predict_proba(eval_X).tolist()
        pred_new = rfc.predict_proba(eval_X_new).tolist()
        pred_y = model.predict(eval_X).tolist()
        pred_y_new = rfc.predict(eval_X_new).tolist()

        test_size = 200000
        for idx in range(test_size):
            label = eval_y.loc[idx]
            if label != pred_y_new[idx] and label == pred_y[idx]:
                print(label, pred[idx], pred_new[idx])
                instance = eval_X_new.loc[[idx]]
                _, bias, contributions = ti.predict(rfc, instance)
                print("Bias", bias)
                print("Feature contributions:")
                for c, feature in zip(contributions[0], plus_feature):
                    print(feature, c)

        # print("union")
        # print(exp.as_list())
        # print("intersection")
        # print(exp_new.as_list())
        """
        cnt1 = 0
        cnt2 = 0
        print(max(diff))
        for idx in range(test_size):
            if pred_y[idx] == pred_y_new[idx]:
                if diff[idx] > 0.01:
                    cnt1 += 1
                if diff[idx] > 0.05:
                    cnt2 += 1
                else:
                    pass
        print(cnt1 / test_size)
        print(cnt2 / test_size)
        """

    def split_dataset(self):
        _, self.eval_X, _, self.eval_y = train_test_split(
            self.X, self.y, test_size=10000, random_state=2019)
        return self.eval_X, self.eval_y

    def loop_split(self, X, y):
        train_X, train_y, test_X, test_y = train_test_split(
            X, y, test_size=20000, random_state=1024)
        return train_X, train_y, test_X, test_y

    def feature_elimination(self):
        model = RandomForestClassifier(n_estimators=100, random_state=2019)
        rfecv = RFECV(model, step=1, cv=10, scoring='f1', n_jobs=3)

        X, eval_X, y, eval_y = train_test_split(
            self.X, self.y, test_size=0.2, random_state=1024)

        for i in range(6):
            X, _, y, _ = self.loop_split(X, y)

        for i in range(7, 11):
            X, train_X, y, train_y = self.loop_split(X, y)
            rfecv.fit(train_X, train_y)
            print("Optimal number of features : %d" % rfecv.n_features_)

            selected_feature = []
            for index in range(len(rfecv.support_)):
                if rfecv.support_[index]:
                    print(self.feature_column[index])
                else:
                    pass

            for imp in rfecv.estimator_.feature_importances_:
                print(imp)

            # Plot number of features VS. cross-validation scores
            plt.figure()
            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score")
            plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
            plt.show()

        return rfecv.support_

    def load_dataset(self):
        self.selected_feature = [
            'ssl_version', 'cipher_suite_server', 'cert_sig_alg',
            'max_duration', 'avg_duration', 'avg_size', 'recv_sent_size_ratio',
            'avg_pkts', 'recv_sent_pkts_ratio', 'avg_time_diff',
            'std_time_diff', 'max_time_diff', 'avg_cert_valid_day',
            'std_cert_valid_day', 'percent_of_valid_cert',
            'avg_valid_cert_percent', 'subject_CN_is_IP', 'subject_only_CN',
            'is_O_in_issuer', 'avg_TTL', 'avg_domain_name_length',
            'std_domain_name_length', 'avg_IPs_in_DNS'
        ]
        self.X = self.X[self.selected_feature]
        self.test_X = self.test_X[self.selected_feature]
        return self.X, self.y, self.test_X, self.test_y


FeatureSelection().interpret()