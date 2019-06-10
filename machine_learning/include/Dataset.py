# -*- coding: utf-8 -*-

from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split


class Dataset():
    def __init__(self, categorical=False):
        # features and labels
        self.CATEGORICAL_COLUMNS = [\
            'ssl_version',\
            'cipher_suite_server',\
            'cert_key_alg',\
            'cert_sig_alg',\
            'cert_key_type',\
        ]
        self.NUMERIC_COLUMNS = [
            'max_duration',\
            'avg_duration',\
            'percent_of_std_duration',\
            'number_of_flows',\
            'ssl_flow_ratio',\
            'avg_size',\
            'recv_sent_size_ratio',\
            'avg_pkts',\
            'recv_sent_pkts_ratio',\
            'packet_loss',\
            'percent_of_established_state',\
            'avg_time_diff',\
            'std_time_diff',\
            'max_time_diff',\
            'ssl_tls_ratio',\
            'resumed',\
            'self_signed_ratio',\
            'avg_key_length',\
            'avg_cert_valid_day',\
            'std_cert_valid_day',\
            'percent_of_valid_cert',\
            'avg_valid_cert_percent',\
            'number_of_cert_serial',\
            'number_of_domains_in_cert',\
            'avg_cert_path',\
            'x509_ssl_ratio',\
            'SNI_ssl_ratio',\
            'is_SNIs_in_SNA_dns',\
            'is_CNs_in_SNA_dns',\
            'subject_CN_is_IP',\
            'subject_is_com',\
            'is_O_in_subject',\
            'is_CO_in_subject',\
            'is_ST_in_subject',\
            'is_L_in_subject',\
            'subject_only_CN',\
            'issuer_is_com',\
            'is_O_in_issuer',\
            'is_CO_in_issuer',\
            'is_ST_in_issuer',\
            'is_L_in_issuer',\
            'issuer_only_CN',\
            'avg_TTL',\
            'avg_domain_name_length',\
            'std_domain_name_length',\
            'avg_IPs_in_DNS']
        self.LABEL_COLUMN = ['label']
        self.USE_COL = self.LABEL_COLUMN + self.CATEGORICAL_COLUMNS + self.NUMERIC_COLUMNS
        self.X = None
        self.y = None
        self.data_size = None
        self.test_X = None
        self.test_y = None
        self.load_dataset(categorical)

    def get_dataset(self):
        return self.X, self.y

    def get_testset(self):
        return self.test_X, self.test_y

    def get_column_name(self):
        return self.CATEGORICAL_COLUMNS + self.NUMERIC_COLUMNS

    def get_feautre_column(self):
        print(self.CATEGORICAL_COLUMNS, self.NUMERIC_COLUMNS)
        return self.CATEGORICAL_COLUMNS, self.NUMERIC_COLUMNS

    def load_dataset(self, categorical=False):
        dataset_path = '../../data_model'
        data_file = tf.gfile.Glob(dataset_path + '*dataset-*.csv')
        if data_file:
            print(">>> Read dataset file")
        else:
            raise FileNotFoundError("No dataset file")

        # create dataset
        train_df = []
        for filename in data_file:
            train_df.append(pd.read_csv(filename, usecols=self.USE_COL))

        dataset = pd.concat(train_df, ignore_index=True)
        self.data_size = dataset.shape[0]

        print(">>> dataset size: ", dataset.shape)

        if not categorical:
            # print(">>> CATEGORICAL_COLUMNS:")
            for key in self.CATEGORICAL_COLUMNS:
                # print(key)
                # string = list(dataset[key].unique())
                dataset[key] = dataset[key].rank(
                    method='dense', ascending=True).astype(int)
                # rank = list(dataset[key].unique())
                # map_list = list(zip(string, rank))
                # for n in map_list:
                #    print("\t%s %s" % (n[0], n[1]))
        else:
            pass

        feature_set = [
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

        dataset = ds.loc[:self.data_size]
        self.y = dataset.pop('label')
        self.X = dataset[feature_set]