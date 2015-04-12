# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series, DataFrame
import matplotlib
from matplotlib import style
style.use('ggplot')
'''
 SVC 简单小例子.
    重点  SVM 的直线结果 各参数解释.
    How svm works
Created on April , 2015
@author: stevey
@python love love love, you can make your best.!!
'''



x = [1,5, 1.5, 8, 1.9]
y = [2, 8, 1.8, 0.6, 11]

# plt.plot(x, y)
# plt.scatter(x, y)

x = np.array([ [1,2],
            [5, 8],
            [1.5, 1.8],
            [8, 8],
            [1, 0.6],
            [9, 11]])
y = [0,1, 0,1,0, 1]
# create classfier
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(x, y)
print(clf.predict([ 0.58, 0.76])) # 0 it locates in small area.

w = clf.coef_[0]
print(w)

# graphs
a = -w[0] / w[1]
xx = np.linspace(0, 12)
yy = a * xx - clf.intercept_[0] / w[1]

h0 = plt.plot(xx, yy, 'k-', label='non weighted div')
plt.scatter(x[:, 0], x[:, 1], c=y)
plt.legend()
plt.show()
