'''
Created on Mar 18, 2022
@author: Ivan Li
'''
# 打印参数
print(__doc__)
# 导入相关的包
import svmSklearn
import svmMLiA

# 加载 linear分类 数据
dataArr, labelsArr = svmMLiA.loadDataSet('testSet.txt')
trainingDataArr = dataArr[:80]
trainingLabelsArr = labelsArr[:80]
testDataArr = dataArr[80:]
testLabelsArr = labelsArr[80:]
# 加载 linear svm
svmSklearn.linear(trainingDataArr, trainingLabelsArr, testDataArr, testLabelsArr)

#加载 RBF分类 数据
trainingDataArr, trainingLabelsArr = svmMLiA.loadDataSet('testSetRBF.txt')
testDataArr, testLabelsArr = svmMLiA.loadDataSet('testSetRBF2.txt')
#加载 RBF svm
svmSklearn.rbf(trainingDataArr, trainingLabelsArr, trainingDataArr, trainingLabelsArr)
svmSklearn.rbf(trainingDataArr, trainingLabelsArr, testDataArr, testLabelsArr)