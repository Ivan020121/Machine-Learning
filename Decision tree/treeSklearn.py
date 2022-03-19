'''
Created on Mar 17, 2022
@author: Ivan Li
'''
# 导入相关的包
from sklearn.tree import DecisionTreeClassifier

def treeSklearn(trainingDataSet, trainingLabelsSet, testDataSet, testLabelsSet):
    # 建立 决策树 分类模型
    clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_features=None,
                                 max_depth=None, min_samples_split=2,min_samples_leaf=1,
                                 min_weight_fraction_leaf=0, max_leaf_nodes=None,
                                 class_weight=None,random_state=None)
    # clf = DecisionTreeClassifier(random_state=0)

    # 打印 相关参数
    print('Parameters: ')
    print(clf)

    # 训练 决策树 分类模型
    clf.fit(trainingDataSet, trainingLabelsSet, sample_weight=None, check_input=True)# X_idx_sorteddeprecated, default=”deprecated”

    # 打印 相关属性
    print('Attributes: ')
    print('The classes labels: ', clf.classes_)
    print('The inferred value of max_features: ', clf.max_features)
    print('The inferred value of max_features: ', clf.n_classes_)
    print('Number of features seen during fit: ', clf.n_features_)
    print('The number of outputs: ', clf.n_outputs_)
    print('The underlying Tree object: ', clf.tree_)

    # 训练集 测试
    # print('Return the depth of the decision tree: ', clf.get_depth())
    print('Return the decision path in the tree: ', clf.decision_path(trainingDataSet, check_input=True))
    print('Return the mean accuracy on the given train data and labels', clf.score(trainingDataSet, trainingLabelsSet, sample_weight=None))

    # 测试集 测试
    print('Return the mean accuracy on the given train data and labels', clf.score(testDataSet, testLabelsSet, sample_weight=None))