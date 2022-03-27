'''
Created on March 27, 2022
@author: Ivan Li
'''

# 导入 相关的包
from apriori_python import apriori

# 加载 数据集
def loadDataSet(filepath):
    dataSet = [line.split() for line in open(filepath).readlines()]
    return dataSet

# apriori-python 1.0.4
def aprioriPy(dataSet, minSup, minConf):
    freqItemSet, rules = apriori(dataSet, minSup, minConf)
    return freqItemSet, rules