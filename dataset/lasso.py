from __future__ import print_function

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
import pylab as pl
pl.figure()
import numpy as np
from sklearn import datasets
diabetes = datasets.load_diabetes()
np.random.seed(1424)
indices = np.random.permutation(len(diabetes.data))
diabetes_X_train = diabetes.data[indices[:-20]]
diabetes_X_test = diabetes.data[indices[-20:]]
diabetes_y_train = diabetes.target[indices[:-20]]
diabetes_y_test = diabetes.target[indices[-20:]]

alphas = np.logspace(-4, -1, 6)
regr = linear_model.Lasso()
scores = [regr.set_params(alpha=alpha).fit(diabetes_X_train, diabetes_y_train).score(diabetes_X_test, diabetes_y_test) for alpha in alphas]

best_alpha = alphas[scores.index(max(scores))]
regr.alpha = best_alpha

print(regr.fit(diabetes_X_train, diabetes_y_train))
print(regr.coef_)