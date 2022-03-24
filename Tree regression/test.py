'''
Created on MArch 24, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import regTrees

# 测试 binSplitDataSet()
testMat = mat(eye(4))
mat0, mat1 = regTrees.binSplitDataSet(testMat, 1, 0.5)
print('mat0: ', mat0)
print('mat1: ', mat1)

# 测试 createTree()
myDat = regTrees.loadDataSet('ex00.txt')
myMat = mat(myDat)
tree = regTrees.createTree(myMat)
print(tree)

myDat1 = regTrees.loadDataSet('ex0.txt')
myMat1 = mat(myDat1)
tree1 = regTrees.createTree(myMat1)
print(tree1)