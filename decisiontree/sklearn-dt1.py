# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey

sklearn example::   Decision Tree
'''

from sklearn import tree
X = [ [0, 0], [1, 1]]
Y = [0, 1]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)

#  other input, give output:: predict target
print(clf.predict([[2, 2]]))

# alternatively,  give the probability of each class
print(clf.predict_proba([[1.7, 2.5]]))







# above:  binary class
# later: multiple classification

# iris

from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)


# after trained, use Graphviz to visualize it.
from sklearn.externals.six import StringIO
with open('iris.dot', 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
import os









# sklearn.tree . DecisionTreeClassifier
# class sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, class_weight=None)[source]¶


## Parameters

### criterion

# source code: https://github.com/scikit-learn/scikit-learn/blob/2b3fabc/sklearn/tree/tree.py#L396


'''
tree 能使用的 几种方法

__all__ = ["DecisionTreeClassifier",
            "DecisionTreeRegressor",
            "ExtraTreeClassifier",
            "ExtraTreeRegressor"]


'''

from sklearn.datasets import load_iris
from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
iris = load_iris()
r = cross_val_score(clf, iris.data, iris.target, cv=10)
print r
''' output
>>> cross_val_score(clf, iris.data, iris.target, cv=10)

array([ 1.        ,  0.93333333,  1.        ,  0.93333333,  0.93333333,
        0.86666667,  0.93333333,  1.        ,  1.        ,  1.        ])
'''