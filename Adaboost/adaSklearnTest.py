'''
Created on March 21, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import adaboostSklearn

# 加载 疝气病马 数据集
trainingDatArr, trainingLabelArr = adaboostSklearn.loadDataSet('horseColicTraining2.txt')
testDatArr, testLabelArr = adaboostSklearn.loadDataSet('horseColicTraining2.txt')

# 加载 AdaBoost 分类器
adaboostSklearn.adaboostSklearn(trainingDatArr, trainingLabelArr, testDatArr, testLabelArr)