'''
Created on March 21, 2022
@author: Ivan Li
'''

# 导入 相关的包
from numpy import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score

# 解析制表符分隔的浮动的常规函数
def loadDataSet(fileName):
    # 获取字段数
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def adaboostSklearn(trainingDataSet, trainingLabelSet, testDataSet, testLabelSet):
    # 建立 adaboost 分类模型
    clf = AdaBoostClassifier(base_estimator=None, n_estimators=1000, learning_rate=1.0, algorithm='SAMME.R', random_state=None)

    # 打印 相关参数
    print('Parameters: ')
    print(clf)

    # 训练 adaboost 训练模型
    clf.fit(trainingDataSet, trainingLabelSet)

    # 打印 相关属性
    print('Attributes: ')
    # print('The base estimator from which the ensemble is grown: ', clf.base_estimator_)
    # print('The collection of fitted sub-estimators: ', clf.estimators_)
    print('The classes labels: ', clf.classes_)
    print('The number of classes: ', clf.n_classes_)
    print('Weights for each estimator in the boosted ensemble: ', clf.estimator_weights_)
    print('Classification error for each estimator in the boosted ensemble: ', clf.estimator_errors_)
    print('The impurity-based feature importances: ', clf.feature_importances_)
    # print('Number of features seen during fit: ', clf.n_features_in_)
    # print('Names of features seen during fit: ', clf.feature_names_in_)

    # 训练集 测试
    train_score = clf.score(trainingDataSet, trainingLabelSet, sample_weight=None)
    print('Return the mean accuracy on the given train data and labels', train_score)

    # 测试集 测试
    test_score = clf.score(testDataSet, testLabelSet, sample_weight=None)
    print('Return the mean accuracy on the given test data and labels', test_score)

    # 交叉验证
    # scores = cross_val_score(clf, dataSet, labelSet, cv=None)
    # return scores