'''
Created on March 28, 2022
@author: Ivan Li
'''

# 导入 相关的包
from fpgrowth_py import fpgrowth

# 加载 数据集
def loadDataSet(filepath):
    dataSet = [line.split() for line in open(filepath).readlines()]
    return dataSet

# fpgrowth-py 1.0.0
def fpGrowthPy(dataSet, minSup, minConf):
    freqItemSet, rules = fpgrowth(dataSet, minSup, minConf)
    return freqItemSet, rules