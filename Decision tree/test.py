'''
Created on Mar 17, 2022
@author: Ivan Li
'''
# 打印参数
print(__doc__)

# 导入相关的包
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import treeSklearn

# 加载 iris 数据集
iris = load_iris()
# 分割数据集
X_train,X_test,y_train,y_test = train_test_split(iris['data'], iris['target'], test_size=0.2, random_state=42)

# 加载 决策树 分类器
treeSklearn.treeSklearn(X_train, y_train, X_test, y_test)