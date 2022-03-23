'''
Created on March 22, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import regressionSklearn

# 加载 数据集
xArr, yArr = regressionSklearn.loadDataSet('abalone.txt')

# 测试 普通最小二乘 线性回归模型
score = regressionSklearn.ordinaryLeastSquares(xArr, yArr)
print('score: ', score)

# 测试 普通最小二乘 线性回归模型
score = regressionSklearn.ridgeRegression(xArr, yArr)
print('score: ', score)

# 测试 Lasso 线性回归模型
scoreLasso, scoreLasIC = regressionSklearn.lassoRegression(xArr, yArr)
print(scoreLasso)
print(scoreLasIC)