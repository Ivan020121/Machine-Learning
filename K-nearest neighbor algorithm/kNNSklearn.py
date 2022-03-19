'''
Created on Mar 19, 2022
@author: Ivan Li
'''
# 导入相关的包
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as kNN

def kNNSklearn(trainingDataSet, trainingLabelsSet, testDataSet, testLabelsSet):
    # 建立 kNN 分类模型
    knn_clf = kNN(n_neighbors=5, weights='distance', algorithm='auto', n_jobs=1)

    # 打印 相关参数
    print(knn_clf)

    # 训练 kNN 分类模型
    knn_clf.fit(trainingDataSet, trainingLabelsSet)

    # 打印 相关属性
    print('Class labels known to the classifier: ', knn_clf.classes_)
    print('The distance metric used: ', knn_clf.effective_metric_)
    print('Additional keyword arguments for the metric function: ', knn_clf.effective_metric_params_)
    # print('Number of features seen during fit: ', knn_clf.n_features_in_)
    # print('Names of features seen during fit: ', knn_clf.features_names_in_)
    # print('Number of samples in the fitted data: ', knn_clf.n_samples_fit_)
    print('outputs_2d_: ', knn_clf.outputs_2d_)

    # 训练集 测试
    print('Return the mean accuracy on the given train data and labels', knn_clf.score(trainingDataSet, trainingLabelsSet, sample_weight=None))

    # 测试集 测试
    print('Return the mean accuracy on the given test data and labels', knn_clf.score(testDataSet, testLabelsSet, sample_weight=None))

