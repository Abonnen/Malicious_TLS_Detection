# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from joblib import load
from time import time
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, auc, roc_curve, confusion_matrix
from lightgbm import LGBMClassifier
from dataset_split import DataSplit

model = load('LGBMClassifier.joblib')
"""
feature_set = [
    'avg_cert_path', 'avg_cert_valid_day', 'avg_domain_name_length',
    'avg_duration', 'avg_IPs_in_DNS', 'avg_pkts', 'avg_size', 'avg_time_diff',
    'avg_TTL', 'avg_valid_cert_percent', 'cert_key_type', 'cert_sig_alg',
    'cipher_suite_server', 'is_CNs_in_SNA_dns', 'is_O_in_issuer',
    'is_O_in_subject', 'is_ST_in_subject', 'max_duration', 'max_time_diff',
    'number_of_domains_in_cert', 'number_of_flows', 'packet_loss',
    'recv_sent_pkts_ratio', 'recv_sent_size_ratio', 'ssl_version',
    'std_domain_name_length', 'std_time_diff', 'subject_only_CN', 'resumed',
    'SNI_ssl_ratio'
]
for i in range(model.n_features_):
    print(feature_set[i], model.feature_importances_[i])
"""
"""
test_m = 5000
test_n = 100000
_, X, _, y = DataSplit().split_dataset(
    test_malicious=test_m, test_normal=test_n)
"""
import sys
sys.path.append("..")
from include.Dataset import Dataset
X, y = Dataset().get_testset()
t0 = time()
y_pred = model.predict(X, num_iteration=model.best_iteration_)
t1 = time()

import sys
import seaborn as sns
sys.path.append("..")
from include.AlarmMetric import false_positive_rate, false_negative_rate, false_discovery_rate
print(">>> predict time: ", t1 - t0)
print("\tACC: %f" % accuracy_score(y, y_pred))
print("\tPPV: %f" % precision_score(y, y_pred))
print("\tTPR: %f" % recall_score(y, y_pred))
print("\tF1: %f" % f1_score(y, y_pred))
print("\tFDR: %f" % false_discovery_rate(y, y_pred))
cm = confusion_matrix(y, y_pred)
sns.set(font_scale=2)
sns.heatmap(cm, annot=True, fmt="d", linewidths=.5, cmap="YlGnBu")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
"""

test_size = len(y)
test_m = sum(y)
N = test_size - test_m

for M in range(1000, test_m + 1, 500):
    X_test = X[0:N + M]
    y_test = y[0:N + M]
    # predict
    print(">>> Test set: M=%d, N=%d" % (M, N))
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    print("\tACC: %f" % accuracy_score(y_test, y_pred))
    print("\tPPV: %f" % precision_score(y_test, y_pred))
    print("\tTPR: %f" % recall_score(y_test, y_pred))
    print("\tF1: %f" % f1_score(y_test, y_pred))
    print("\tFPR: %f" % false_positive_rate(y_test, y_pred))
    print("\tFNR: %f" % false_negative_rate(y_test, y_pred))
    print("\tFDR: %f" % false_discovery_rate(y_test, y_pred))

    # ROC curve
    # FPR = false positive rate, TPR = true positive rate
    predict_probaY = model.predict_proba(X_test)
    prediction = predict_probaY[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, prediction)
    roc_auc = auc(fpr, tpr)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.5f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
"""
"""
y_pred = model.predict(X, num_iteration=model.best_iteration_)

from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y, y_pred)

from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.utils.fixes import signature

precision, recall, _ = precision_recall_curve(y, y_pred)

print('Average precision-recall score: {0:0.2f}'.format(average_precision))
# In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
step_kwargs = ({
    'step': 'post'
} if 'step' in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color='b', alpha=0.1, where='post')
plt.fill_between(recall, precision, alpha=0.1, color='b', **step_kwargs)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title(
    'Precision-Recall curve (GBDT): AP={0:0.2f}'.format(average_precision))
plt.show()
"""
