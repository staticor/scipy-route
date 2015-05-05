# -*- coding: utf-8 -*-

'''
Created on May , 2015
@author: stevey
'''

import numpy as np
np.random.seed(1428)
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import os, sys
from sklearn import tree
style.use('ggplot')


X = [[0, 0], [2, 2]]
y = [0.5, 2.5]

clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
print(clf.predict([[1, 1]]))
# array([0.5])

def oye():
    print('o'*79)
oye()
# example:  plot_tree_regression.py
# http://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#example-tree-plot-tree-regression-py


from sklearn.tree import DecisionTreeRegressor


# create a random dataset

rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))

# Fit regression model

clf_1 = DecisionTreeRegressor(max_depth=2)
clf_2 = DecisionTreeRegressor(max_depth=5)
clf_1.fit(X, y)
clf_2.fit(X, y)

# Predict

X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = clf_1.predict(X_test)
y_2 = clf_2.predict(X_test)

# Plot the results

plt.figure()
plt.scatter(X, y , c='k', label='data')
plt.plot(X_test, y_1, c='g', label='max_dept=2', linewidth=2)
plt.plot(X_test, y_2, c='r', label='max_dept=5', linewidth=2)
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regression')

plt.legend()
plt.show()