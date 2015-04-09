# -*- coding: utf-8 -*-
"""
Created on Thu Apr 09 09:32:12 2015
'life is short, python is awesome!'
@author: STEVE
"""

import matplotlib.pyplot as plt 
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()

# check the source data's volume
print('We have total about {count} rows of data'.format(count=len(digits.data)))


clf = svm.SVC(gamma=0.001, C=100)
x, y = digits.data[:-1], digits.target[:-1]
clf.fit(x, y)

print('Predictions:', clf.predict(digits.data[-3]))

# show the real digits picture
plt.imshow(digits.images[-3], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()