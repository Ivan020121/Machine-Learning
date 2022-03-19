'''
Created on Mar 19, 2022
@author: Ivan Li
'''
# 打印 参数
print(__doc__)

import kNN
from numpy import *
import matplotlib.pyplot as plt
import kNNSklearn

# group, labels = kNN.createDataSet()
# print(group)
# print(labels)
#
# result0 = kNN.classify0([0, 0], group, labels, 3)
# print(result0)

datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')
print(datingDataMat)
print(datingLabels)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:, 0], datingDataMat[:, 1], 15.0*array(datingLabels), 15.0*array(datingLabels))
plt.show()

normMat, ranges, minVals = kNN.autoNorm(datingDataMat)
print(normMat)
print(ranges)
print(minVals)

kNNSklearn.kNNSklearn(normMat[:800], datingLabels[:800], normMat[800:], datingLabels[800:])

# result1 = kNN.datingClassTest()
# print(result1)
#
# result2 = kNN.handwritingClassTest()
# print(result2)