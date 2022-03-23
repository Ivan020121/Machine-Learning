'''
Created on March 22, 2022
@author: Ivan Li
'''

# 导入 相关的包
from numpy import *
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.model_selection import cross_val_score

# 解析制表符分隔的浮动的常规函数
def loadDataSet(fileName):
    # 获取字段数
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

# 普通最小二乘
def ordinaryLeastSquares(X_set, y_set):
    # 建立 普通最小二乘 线性回归模型
    reg = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=None)

    # 打印 相关参数
    print('Parameters: ')
    print(reg)

    # 训练 普通最小二乘 线性回归模型
    reg.fit(X_set, y_set, sample_weight=None)

    # 打印 相关属性
    print('Attributes: ')
    print('Estimated coefficients for the linear regression problem: ', reg.coef_)
    print('Rank of matrix X: ', reg.rank_)
    print('Singular values of X: ', reg.singular_)
    print('Independent term in the linear model: ', reg.intercept_)
    # print('Number of features seen during fit: ', reg.n_features_in_)
    # print('Names of features seen during fit: ', reg.feature_names_in_)

    # 交叉验证
    score = cross_val_score(reg, X_set, y_set, cv=None)
    return score

# 岭回归
def ridgeRegression(X_set, y_set):
    # 建立 岭回归 回归模型
    rng = Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True,
                max_iter=None, tol=1e-3, solver='auto', random_state=None)

    # 打印 相关参数
    print('Parameters: ')
    print(rng)

    # 数据 标准化处理
    scaler = StandardScaler()
    XCopy = scaler.fit_transform(X_set, y=None)

    # 训练 岭回归 回归模型
    rng.fit(XCopy, y_set, sample_weight=None)

    # 打印 相关属性
    print('Attributes: ')
    print('Weight vector(s): ', rng.coef_)
    print('Independent term in decision function: ', rng.intercept_)
    print('Actual number of iterations for each target: ', rng.n_iter_)
    # print('Number of features seen during fit: ', rng.n_features_in_)
    # print('Names of features seen during fit: ', rng.feature_names_in_)

    # 交叉验证
    score = cross_val_score(rng, X_set, y_set, cv=None)
    # score = rng.score(XCopy, y_set)
    return score

# Lasso回归
def lassoRegression(X_set, y_set):
    # 建立 Lasso 回归模型
    clf = Lasso(alpha=0.1, fit_intercept=True, normalize=False, precompute=False,
                copy_X=True, max_iter=1000, tol=1e-4, warm_start=False,
                positive=False, random_state=None, selection='cyclic')
    reg = LassoLarsIC(criterion='aic', normalize=False)
    # clf = Lasso(alpha=0.1)

    # 打印 相关参数
    print('Parameters: ')
    print(clf)

    # 数据 标准化处理
    scaler = StandardScaler()
    XCopy = scaler.fit_transform(X_set, y=None)

    # 训练 Lasso 回归模型
    clf.fit(XCopy, y_set, check_input=True)
    reg.fit(XCopy, y_set)

    # 打印 相关属性
    print('Attributes: ')
    print('Parameter vector (w in the cost function formula): ', clf.coef_)
    print('Given param alpha, the dual gaps at the end of the optimization, same shape as each observation of y: ', clf.dual_gap_)
    print('Sparse representation of the fitted coef_: ', clf.sparse_coef_)
    print('Independent term in decision function: ', clf.intercept_)
    print('Number of iterations run by the coordinate descent solver to reach the specified tolerance: ', clf.n_iter_)
    # print('Number of features seen during fit: ', clf.n_features_in_)
    # print('Names of features seen during fit: ', rng.feature_names_in_)

    # 交叉验证
    scoreLasso = cross_val_score(clf, XCopy, y_set, cv=None)
    scoreLasIC = cross_val_score(reg, XCopy, y_set, cv=None)
    return scoreLasso, scoreLasIC