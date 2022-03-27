'''
Created on March 27, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
import apriori

dataSet = apriori.loadDataSet()
print(dataSet)
C1 = apriori.createC1(dataSet)
print(C1)
print(dataSet)
L1, suppData0 = apriori.scanD(dataSet, C1, 0.5)
print(L1)
L, suppData = apriori.apriori(dataSet)
print(L)
print(apriori.aprioriGen(L[0], 2))

rules = apriori.generateRules(L, suppData, 0.7)
print(rules)