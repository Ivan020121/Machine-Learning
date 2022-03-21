'''
Created on March 21, 2022
@author: Ivan Li
'''

# 导入 相关的包
from numpy import *

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

# 数据分类
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray
    

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    # 始误差和，至无穷大
    minError = inf
    # 在所有维度上循环
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        # 循环当前维度中的所有范围
        for j in range(-1,int(numSteps)+1):
            # 大于或小于
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                # 用i，j，lessThan给stump分类
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                print("predictedVals",predictedVals.T,"errArr",errArr.T)
                # 计算总误差乘以D
                weightedError = D.T*errArr
                print("split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrain(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    # 初始化 D
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        #print "error",error
        # 计算alpha，加入max（错误，每股收益）以说明错误=0
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha  
        print("alpha",alpha)
        weakClassArr.append(bestStump)
        print("classEst",classEst)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        print("D",D)
        # 所有分类器的计算训练错误，如果为0，则提前退出循环（使用中断）
        aggClassEst += alpha*classEst
        print("aggClassEst",aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        #print aggErrors
        errorRate = aggErrors.sum()/m
        print(errorRate)
        if errorRate == 0.0: break
    return weakClassArr