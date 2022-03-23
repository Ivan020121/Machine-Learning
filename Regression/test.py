'''
Created on March 22, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import matplotlib.pyplot as plt
import regression

# 加载 数据集
xArr, yArr = regression.loadDataSet('ex0.txt')

# 测试 standRegres()
ws = regression.standRegres(xArr, yArr)
print('ws: ', ws)

# 绘制 数据集散点图 最佳拟合直线图
xMat = mat(xArr)
yMat = mat(yArr)
yHat = xMat*ws
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
xCopy = xMat.copy()
xCopy.sort(0)
yHat = xCopy*ws
ax.plot(xCopy[:, 1], yHat)
plt.show()

# 计算 相关系数
print(corrcoef(yHat.transpose(), yMat))

# 点估计
print(yArr[0])
print(regression.lwlr(xArr[0], xArr, yArr, 1.0))
print(regression.lwlr(xArr[0], xArr, yArr, 0.001))
yHat = regression.lwlrTest(xArr, xArr, yArr, 0.003)

# 绘制 数据集散点图 最佳拟合直线图
xMat = mat(xArr)
srtInd = xMat[:, 1].argsort(0)
xSort = xMat[srtInd][:, 0, :]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(xSort[:, 1], yHat[srtInd])
ax.scatter(xMat[:, 1].flatten().A[0], mat(yArr).T.flatten().A[0], s = 2, c = 'red')
plt.show()