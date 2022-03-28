'''
Created on March 28, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
import fpGrowthPy

# 加载 新闻网站点击流 数据集
itemList = fpGrowthPy.loadDataSet('kosarak.dat')
# print(itemList)

# fpgrowth-py 1.0.0
freqItemSet, rules = fpGrowthPy.fpGrowthPy(itemList, float(100000)/len(itemList), 0.5)
print(freqItemSet)