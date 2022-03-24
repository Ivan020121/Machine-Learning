'''
Created on MArch 24, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import regTreesSklearn

# 加载 ex00.txt 数据集
myDat = regTreesSklearn.loadDataSet('ex00.txt')
myMat = mat(myDat)
# 加载 回归树 回归模型
m ,n = shape(myMat)
# print(myMat[:, n-2])
# print(myMat[:, -1])
score = regTreesSklearn.createTree(myMat[:, n-2], myMat[:, -1], 4)
print(score)

# 加载 ex0.txt 数据集
myDat = regTreesSklearn.loadDataSet('ex0.txt')
myMat = mat(myDat)
# 加载 回归树 回归模型
m, n = shape(myMat)
# print(myMat[:, n-2])
# print(myMat[:, -1])
score = regTreesSklearn.createTree(myMat[:, n-2], myMat[:, -1], 4)
print(score)