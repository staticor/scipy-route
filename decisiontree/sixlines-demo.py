# coding: utf-8

## classification
## DecisionTreeClassifier for binary classification

from sklearn import tree
p1 = [0, 0]
p2 = [1, 1]
Y = [0, 1]
X = [p1, p2]
clf = tree.DecisionTreeClassifier()
clf.fit(X, Y)
predict = [2, 2]
print clf.predict([predict ])

## DecisionTreeClassifier for multi-classification


from sklearn import tree
from sklearn.datasets import load_iris

iris = load_iris()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)