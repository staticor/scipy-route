# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series, DataFrame
'''
Created on April , 2015
@author: stevey
@python love love love, you can make your best.!!
'''

from sklearn import tree
from sklearn.externals.six import StringIO

from sklearn.datasets import load_iris

iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
import os
print(os.getcwd())
with open('iris_mac.dot', 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

# os.unlink('iris_mac.dot')
