# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey

'''

import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

y_categories = np.unique(iris_y)
print(y_categories) # Three species of Iris

# classifier: K-NN (K-Nearest Neighbors)

# Split iris data --> train and test data

#  by np's random permutation to split
np.random.seed(828)
indices = np.random.permutation(len(iris_X))
# print(indices)

# We set top-40 to be in train set.  Final 10 to be tested
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]

# Create the knn classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',  metric_params=None, n_neighbors=5, p=2, weights='uniform')
knn.fit(iris_X_train, iris_y_train)
predict_rest = knn.predict(iris_X_test)
# show rest
for i in range(10):
    real=iris_y_test[i]
    predict=predict_rest[i]
    is_equal = (real != predict) * ' '* 4
    print('{p}real: {real} \t predict: {predict}'.format(p=is_equal,real=real, predict=predict))

# The result showed the 3rd case is not same.


# Diabetes dataset
# Prepare for modeling -- splitting dataset
diabetes = datasets.load_diabetes()
np.random.seed(1424)
indices = np.random.permutation(len(diabetes.data))
diabetes_X_train = diabetes.data[indices[:-20]]
diabetes_X_test = diabetes.data[indices[-20:]]
diabetes_y_train = diabetes.target[indices[:-20]]
diabetes_y_test = diabetes.target[indices[-20:]]

# print(diabetes_X_test, len(diabetes_X_test))

from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
#LinearRegression(copy_X=True, fit_intercept=True, normalize=False)
print(regr.coef_)

# The mean square error
print(np.mean( (regr.predict(diabetes_X_test) - diabetes_y_test )**2 ))
# MSE: 2413.65599655
# Explained Variance Score: 1 is the perfect prediction, and 0 is worst -> There is no linear relationship -> You choose wrong estimator.

print(regr.score(diabetes_X_test, diabetes_y_test))
# 0.43

x_indices = np.arange(len(diabetes_X_test))

import matplotlib.pyplot as plt
plt.scatter(x_indices, regr.predict(diabetes_X_test), color='g')
plt.scatter(x_indices, diabetes_y_test, color='r')

regr = linear_model.Ridge(alpha=.1)

np.random.seed(828)
for _ in range(6):
    this_X = .1 * np.random.normal(size=(2, 1)) + X
plt.show()


