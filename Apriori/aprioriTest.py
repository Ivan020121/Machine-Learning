'''
Created on Maarch 25, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
import aprioriPy
from numpy import *

# 加载 毒蘑菇 数据集
itemList = aprioriPy.loadDataSet('mushroom.dat')
# print(itemList)

# apriori-python 1.0.4
freqItemSet, rules = aprioriPy.aprioriPy(itemList, 0.3, 0.5)
for item in freqItemSet[2]:
    if '2' in item:
        print(item)
# print(rules)