from sklearn import svm
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
svc = svm.SVC(kernel='linear')
print(svc.fit(iris_X_train, iris_y_train))
