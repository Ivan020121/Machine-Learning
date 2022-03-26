'''
Created on March 25, 2022
@author: Ivan Li
'''

# 导入 相关的包
import matplotlib
import matplotlib.pyplot as plt
from numpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering

# 加载 地理坐标 数据集
def loadDataSet():
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    return datMat

# 数据 标准化 处理
def standardize(dataSet):
    X = StandardScaler().fit_transform(dataSet)
    return X

# K均值 聚类模型
def kMeans(dataSet, numClust):
    # 建立 K均值 聚类模型
    kmeans = KMeans(n_clusters=numClust, init='k-means++',
                    n_init=10, max_iter=1000, tol=1e-4, verbose=0,
                    random_state=0, copy_x=True, algorithm='elkan')

    # 打印 相关参数
    print('Parameters: ')
    print(kmeans)

    # 训练 K均值 聚类模型
    kmeans.fit(dataSet, y=None)

    # 打印 相关属性
    print('Attributes: ')
    # print()

    # 测试
    centroids = kmeans.cluster_centers_
    return centroids

# 高斯混合 聚类模型
def gaussianMixture(dataSet, numClust):
    # 建立 高斯混合 聚类模型
    gm = GaussianMixture(n_components=numClust, covariance_type='full', tol=1e-3, reg_covar=0,
                         max_iter=100, n_init=1, init_params='kmeans', weights_init=None, means_init=None,
                         precisions_init=None, random_state=0, warm_start=False, verbose=0, verbose_interval=10)

    # 打印 相关参数
    print('Parameters: ')
    print(gm)

    # 训练 高斯混合 聚类模型
    gm.fit(dataSet, y=None)

    # 打印 相关属性
    print('Attributes: ')
    print('The weights of each mixture components: ', gm.weights_)
    print('The mean of each mixture component: ', gm.means_)
    print('The covariance of each mixture component: ', gm.covariances_)
    print('The precision matrices for each component in the mixture: ', gm.precisions_)
    print('The cholesky decomposition of the precision matrices of each mixture component: ', gm.precisions_cholesky_)
    print('True when convergence was reached in fit(), False otherwise: ', gm.converged_)
    print('Number of step used by the best fit of EM to reach the convergence: ', gm.n_iter_)
    print('Lower bound value on the log-likelihood of the best fit of EM: ', gm.lower_bound_)

    # 测试
    means = gm.means_
    return means

# DBSCAN 聚类模型
def dbscan(dataSet):
    # 建立 DBSCAN 聚类模型
    clustering = DBSCAN(eps=0.5, min_samples=3, metric='euclidean',
                        metric_params=None, algorithm='auto', p=None, n_jobs=None)

    # 打印 相关参数
    print('Parameters: ')
    print(clustering)

    # 训练 DBSCAN 聚类模型
    clustering.fit(dataSet, y=None)

    # 打印 相关属性
    print('Attributes: ')
    print('Indices of core samples: ', clustering.core_sample_indices_)
    print('Copy of each core sample found by training: ', clustering.components_)
    # print('Number of features seen during fit: ', clustering.n_features_in_)
    # print('Number of features seen during fit: ', clustering.feature_names_in_)

    # 测试
    labels = clustering.labels_
    return labels

# AGNES 聚类模型
def agnes(dataSet, numClust):
    # 建立 AGNES 聚类模型
    clustering = AgglomerativeClustering(n_clusters=numClust, affinity='euclidean', memory=None,
                                         connectivity=None, compute_full_tree='auto', linkage='ward')

    # 打印 相关参数
    print('Parameters: ')
    print(clustering)

    # 训练 AGNES 聚类模型
    clustering.fit(dataSet, y=None)

    # 打印 相关属性
    print('Attributes: ')
    print('The number of clusters found by the algorithm: ', clustering.n_clusters)
    print('Cluster labels for each point: ', clustering.labels_)
    print('Number of leaves in the hierarchical tree: ', clustering.n_leaves_)
    # print('Number of features seen during fit: ', clustering.n_features_in_)
    # print('Number of features seen during fit: ', clustering.feature_names_in_)

    # 测试
    labels = clustering.labels_
    return labels

# 绘制 簇聚类
def plotCluster(myCentroids):
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()