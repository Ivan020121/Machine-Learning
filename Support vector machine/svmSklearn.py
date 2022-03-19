'''
Created on Mar 18, 2022
@author: Ivan Li
'''
'''
SVC参数解释

（1）C: 目标函数的惩罚系数C，用来平衡分类间隔margin和错分样本的，default C = 1.0；

（2）kernel：参数选择有RBF, Linear, Poly, Sigmoid, 默认的是"RBF";

（3）degree：if you choose 'Poly' in param 2, this is effective, degree决定了多项式的最高次幂；

（4）gamma：核函数的系数('Poly', 'RBF' and 'Sigmoid'), 默认是gamma = 1 / n_features;

（5）coef0：核函数中的独立项，'RBF' and 'Poly'有效；

（6）probablity: 可能性估计是否使用(true or false)；

（7）shrinking：是否进行启发式；

（8）tol（default = 1e - 3）: svm结束标准的精度;

（9）cache_size: 制定训练所需要的内存（以MB为单位）；

（10）class_weight: 每个类所占据的权重，不同的类设置不同的惩罚参数C, 缺省的话自适应；

（11）verbose: 跟多线程有关，不大明白啥意思具体；

（12）max_iter: 最大迭代次数，default = 1， if max_iter = -1, no limited;

（13）decision_function_shape ： ‘ovo’ 一对一, ‘ovr’ 多对多 or None 无, default=None

（14）random_state ：用于概率估计的数据重排时的伪随机数生成器的种子。
'''
print(__doc__)
# 导入相关的包
import numpy as np
# 绘图功能
import pylab as pl
# sklearn 库中导入 svm 模块
from sklearn.svm import SVC

def linear(trainingDataSet, trainingLabelsSet, testDataSet, testLabelsData):
    # 建立 线性svm 模型
    clf = SVC(kernel='linear')
    clf.fit(trainingDataSet, trainingLabelsSet)
    # 获得划分超平面
    # 划分超平面原方程：w0x0 + w1x1 + b = 0
    # 将其转化为点斜式方程，并把 x0 看作 x，x1 看作 y，b 看作 w2
    # 点斜式：y = -(w0/w1)x - (w2/w1)
    w = clf.coef_[0]  # w 是一个二维数据，coef 就是 w = [w0,w1]
    a = -w[0] / w[1]  # 斜率
    xx = np.linspace(-2, 12)  # 从 -5 到 5 产生一些连续的值（随机的）
    # .intercept[0] 获得 bias，即 b 的值，b / w[1] 是截距
    yy = a * xx - (clf.intercept_[0]) / w[1]  # 带入 x 的值，获得直线方程

    # 画出和划分超平面平行且经过支持向量的两条线（斜率相同，截距不同）
    b = clf.support_vectors_[0]  # 取出第一个支持向量点
    yy_down = a * xx + (b[1] - a * b[0])
    b = clf.support_vectors_[-1]  # 取出最后一个支持向量点
    yy_up = a * xx + (b[1] - a * b[0])

    # 查看相关的参数值
    print("w: ", w)
    print("a: ", a)
    print("support_vectors_: ", clf.support_vectors_)
    print("clf.coef_: ", clf.coef_)

    # 绘制划分超平面，边际平面和样本点
    pl.plot(xx, yy, 'k-')
    pl.plot(xx, yy_down, 'k--')
    pl.plot(xx, yy_up, 'k--')
    # 圈出支持向量
    pl.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
               s=80, facecolors='none')
    X = np.array(trainingDataSet)
    Y = np.array(trainingLabelsSet)
    pl.scatter(X[:, 0], X[:, 1], c=Y, cmap=pl.cm.Paired)

    pl.axis('tight')
    pl.show()

    # 测试集 检验
    errorCount = 0
    m, n = np.shape(testDataSet)
    for i in range(m):
        predict = clf.predict([testDataSet[i]])
        if np.sign(predict) != np.sign(testLabelsData[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount)/m))

def rbf(trainingDataSet, trainingLabelsSet, testDataSet, testLabelsData):
    # 建立 高斯核svm 模型
    clf = SVC(C=1.21, gamma=0.6, tol=1e-10, max_iter=-1, probability=True)
    clf.fit(trainingDataSet, trainingLabelsSet)

    # 查看相关的参数值
    # clf相关参数
    print(clf)
    # 支持向量
    print("support_vectors_: ", clf.support_vectors_)
    # 属于支持向量的点的 index
    print("index_support_vectors_:", clf.support_)
    # 在每一个类中有多少个点属于支持向量
    print("number_support_vectors_:", clf.n_support_)

    # 测试集 检验
    errorCount = 0
    m, n = np.shape(testDataSet)
    for i in range(m):
        predict = clf.predict([testDataSet[i]])
        if np.sign(predict) != np.sign(testLabelsData[i]):
            errorCount += 1
    print("the test error rate is: %f" % (float(errorCount) / m))