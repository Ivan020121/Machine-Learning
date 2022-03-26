'''
Created on March 25, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
from numpy import *
import kMeansSklearn

# 加载 地理坐标 数据集
myData = kMeansSklearn.loadDataSet()

# 测试 K均值 聚类模型
myCentroids = kMeansSklearn.kMeans(myData, 5)
print(myCentroids)
kMeansSklearn.plotCluster(mat(myCentroids))

# 测试 高斯混和 聚类算法
myMeans = kMeansSklearn.gaussianMixture(myData, 5)
print(myMeans)
kMeansSklearn.plotCluster(mat(myMeans))

# 测试 DBSCAN 聚类算法
myData = kMeansSklearn.standardize(myData)
myLabels = kMeansSklearn.dbscan(myData)
print(myLabels)

# 测试 AGNES 聚类算法
myData = kMeansSklearn.standardize(myData)
myLabels = kMeansSklearn.agnes(myData, 5)
print(myLabels)
# [0 0 0 0 0 0 2 2 3 3 0 0 0 1 2 0 0 0 4 2 0 1 0 2 2 0 1 4 4 4 2 0 0 2 4 2 4 2 1 0 3 0 0 2 2 0 2 0 1 2 4 3 4 0 1 0 0 0 2 2 2 0 4 0 2 0 4 3 3]