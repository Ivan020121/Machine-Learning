'''
Created on March 20, 2022
@author: Ivan Li
'''
# 导入相关的包
import re
from numpy import *
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# 字符串 解析
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

# 创建 词汇列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #union of the two sets
    return list(vocabSet)

# 单词 to 向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word: %s is not in my Vocabulary!" % word)
    return returnVec

# 创建 数据集
def createDataSet():
    docList = []
    classList = []
    fullText = []
    dataSet = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    for data in docList:
        dataSet.append(setOfWords2Vec(vocabList, data))
    X_train, X_test, y_train, y_test = train_test_split(dataSet, classList, test_size=0.2, random_state=42)
    return  X_train, X_test, y_train, y_test

def bayesSklearn(trainingDataSet, trainingClassSet, testDataSet, testClassSet):
    # 建立 朴素贝叶斯 分类模型
    clf = GaussianNB()

    # 打印 相关参数
    print('Parameters: ')
    print(clf)

    # 训练 朴素贝叶斯 分类模型
    clf.fit(trainingDataSet, trainingClassSet)

    # 打印 相关属性
    print('Attributes: ')
    print('number of training samples observed in each class: ', clf.class_count_)
    print('probability of each class: ', clf.class_prior_)
    print('class labels known to the classifier: ', clf.classes_)
    # print('absolute additive value to variances: ', clf.epsilon_)
    # print('number of features seen during fit: ', clf.n_features_in_)
    # print('names of features seen during fit: ', clf.feature_names_in)
    print('DEPRECATED: ', clf.sigma_)
    # print('variance of each feature per class: ', clf.var_)
    print('mean of each feature per class: ', clf.theta_)

    # 训练集 测试
    print('Return the mean accuracy on the given train data and labels',clf.score(trainingDataSet, trainingClassSet, sample_weight=None))

    # 测试集 测试
    print('Return the mean accuracy on the given train data and labels',clf.score(testDataSet, testClassSet, sample_weight=None))