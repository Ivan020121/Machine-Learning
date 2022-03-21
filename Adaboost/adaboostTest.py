'''
Created on March 21, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import adaboost

# 测试 简单数据集
datMat, classLabels = adaboost.loadSimpData()
# 设置 样本权重
D = mat(ones((5, 1))/5)
# 加载 单层决策树 分类器
bestStump, minError, bestClasEst = adaboost.buildStump(datMat, classLabels, D)
print(bestStump)
print(minError)
print(bestClasEst)
# 加载 AdaBoost 分类器
classifierArr = adaboost.adaBoostTrainDS(datMat, classLabels, 30)

# 测试 困难数据集
datArr, labelArr = adaboost.loadDataSet('horseColicTraining2.txt')
# 加载 AdaBoost 分类器
classifierArr = adaboost.adaBoostTrainDS(datArr, labelArr, 10)
# 测试 困难数据集 测试集
testArr, testLabelArr = adaboost.loadDataSet('horseColicTest2.txt')
# 加载 AdaBoost 分类器
prediction10 = adaboost.adaClassify(testArr, classifierArr)
errArr = mat(ones((67, 1)))
print(errArr[prediction10 != mat(testLabelArr).T].sum())
