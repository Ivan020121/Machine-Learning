'''
Created on March 20, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
import numpy as np
import bayesSklearn

# 加载 邮件分类 数据集
trainingEmailSet, testEmailSet, trainingClassSet, testClassSet = bayesSklearn.createDataSet()

# 加载 朴素贝叶斯 分类模型
bayesSklearn.bayesSklearn(np.array(trainingEmailSet), np.array(trainingClassSet), np.array(testEmailSet), np.array(testClassSet))