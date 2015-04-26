# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''


from sklearn import datasets

iris = datasets.load_iris()
digits = datasets.load_digits()
# show digits' data and shape
print(digits.data)
print(len(digits.data[0]), len(digits.data))
print(digits.target, len(digits.target))

from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100)
clf.fit(digits.data[:-1], digits.target[:-1])
print('predict result: {ele}'.format(ele=clf.predict(digits.data[-1])))
import matplotlib.pyplot as plt
img = digits.data[-1].reshape(8, 8)
plt.imshow(img)
plt.show()




clf = svm.SVC()
# print(iris)
X, y = iris.data, iris.target
clf.fit(X, y)

import pickle
# save former clf (model)
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0]))



