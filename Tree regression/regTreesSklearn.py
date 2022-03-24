'''
Created on MArch 24, 2022
@author: Ivan Li
'''

# 导入 相关的包
from numpy import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# 解析制表符分隔的浮动的常规函数
def loadDataSet(fileName):
    # 假设最后一列是目标值
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将所有元素映射到float()
        fltLine = list(map(float,curLine))
        dataMat.append(fltLine)
    return dataMat

def createTree(X_set, y_set, tolN):
    # 建立 回归树 回归模型
    regressor = DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None,
                                      min_samples_split=tolN, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                      max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0)

    # 打印 相关参数
    print('Parameters: ')
    print(regressor)

    # 训练 回归树 回归模型
    regressor.fit(X_set, y_set, sample_weight=None)

    # 打印 相关属性
    print('Attributes: ')
    print('Return the feature importances: ', regressor.feature_importances_)
    print('The inferred value of max_features: ', regressor.max_features)
    print('Number of features seen during fit: ', regressor.n_features_)
    # print('Names of features seen during fit: ', regressor.n_features_in_)
    print('The number of outputs when fit is performed: ', regressor.n_outputs_)
    print('The underlying Tree object: ', regressor.tree_)

    # 交叉验证
    score = cross_val_score(regressor, X_set, y_set, cv=10)
    return score