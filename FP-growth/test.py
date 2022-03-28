'''
Created on March 28, 2022
@author: Ivan Li
'''

# 打印 参数
print(__doc__)

# 导入 相关的包
import fpGrowth

# 创建FP树的数据结构
rootNode = fpGrowth.treeNode('pyramid', 9, None)# 创建树中的一个单节点
rootNode.children['eye'] = fpGrowth.treeNode('eye', 13, None)# 增加一个子节点
rootNode.disp()# 显示节点
rootNode.children['phoenix'] = fpGrowth.treeNode('phoenix', 3, None)# 增加一个子节点
rootNode.disp()# 显示节点

# FP树构建函数
simpDat = fpGrowth.loadSimpDat()# 导入数据实例
# print(simpDat)
initSet = fpGrowth.createInitSet(simpDat)# 格式化处理
# print(initSet)
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 3)# 创建FP树
myFPtree.disp()# 文本表示结果

# 从新闻网站点击流中挖掘
parasedDat = [line.split() for line in open('kosarak.dat').readlines()]# 将数据集导入列表
# print(parasedDat)
initSet = fpGrowth.createInitSet(parasedDat)# 初始集合初始化
myFPtree, myHeaderTab = fpGrowth.createTree(initSet, 100000)# 构建FP树
myFreqList = []# 创建空列表保存频繁项集
fpGrowth.mineTree(myFPtree, myHeaderTab, 100000, set([]), myFreqList)# 结果
print(myFreqList)