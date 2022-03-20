'''
Created on March 20, 2022
@author: Ivan Li
'''
# 导入 相关的包
import numpy as np
from sklearn.linear_model import LogisticRegression

# 处理 数据集
def createDataSet():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    testSet = []; testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))
    return trainingSet, trainingLabels, testSet, testLabels

def logRegresSklearn(trainingDataSet, trainingLabelsSet, testDataSet, testLabelsSet):
    # 获得 最优迭代次数
    best_score = 0
    best_iter = 0

    # 建立 logistic回归 分类模型
    for i in range(1, 1000):
        clf = LogisticRegression(penalty='l2', solver='saga', max_iter = i)

        # 打印 相关参数
        print('Parameters: ')
        print(clf)

        # 训练 logistic回归 分类模型
        clf.fit(trainingDataSet, trainingLabelsSet)

        # 打印 相关属性
        print('Attributes: ')
        print('A list of class labels known to the classifier: ', clf.classes_)
        print('Coefficient of the features in the decision function: ', clf.coef_)
        print('Intercept (a.k.a. bias) added to the decision function: ', clf.intercept_)
        # print('Number of features seen during fit: ', clf.n_features_in_)
        # print('Names of features seen during fit: ', clf.feature_names_in_)
        print('Actual number of iterations for all classes: ', clf.n_iter_)

        # 训练集 测试
        train_score = clf.score(trainingDataSet, trainingLabelsSet, sample_weight=None)
        # print('Return the mean accuracy on the given train data and labels', train_score)

        # 测试集 测试
        test_score = clf.score(testDataSet, testLabelsSet, sample_weight=None)
        # print('Return the mean accuracy on the given test data and labels', test_score)

        aver_score = (test_score+train_score)/2
        if best_score < aver_score:
            best_score = aver_score
            best_iter = i
    return best_score, best_iter