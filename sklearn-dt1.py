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
# create a pdf = file