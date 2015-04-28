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





clf = svm.SVC()
# print(iris)
X, y = iris.data, iris.target
clf.fit(X, y)

import pickle
# save former clf (model)
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
print(clf2.predict(X[0]))


from sklearn import datasets
diabetes = datasets.load_diabetes()

d_data = diabetes.data
d_y = diabetes.target
print(d_data.shape)
print(d_data[1])

