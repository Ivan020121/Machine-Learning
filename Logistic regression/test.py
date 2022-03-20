'''
Created on March 20, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import logRegresSklearn

# 加载 疝气病马 数据集
trainingDataSet, trainingLabelsSet, testDataSet, testLabelsSet = logRegresSklearn.createDataSet()

# 加载 logistic回归 分类模型
best_score, best_iter = logRegresSklearn.logRegresSklearn(trainingDataSet, trainingLabelsSet, testDataSet, testLabelsSet)

# 打印 最优结果
print('Best score: ', best_score)
print('Best iter:', best_iter)