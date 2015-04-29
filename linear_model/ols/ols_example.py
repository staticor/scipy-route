# -*- coding: utf-8 -*-
'''
Created on April , 2015
link: http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html
@author: stevey
'''

import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt

from sklearn import datasets, linear_model
style.use('ggplot')


# load data
diabetes = datasets.load_diabetes()
# select feature to analysis
diabetes_X = np.array(diabetes.data[:, np.newaxis])
diabetes_X = diabetes_X[:, :, 2]
dx_train = diabetes_X[:-20]
dx_test = diabetes_X[-20:]
dy_train = diabetes.target[:-20]
dy_test = diabetes.target[-20:]

regr = linear_model.LinearRegression()
regr.fit(dx_train, dy_train)

# output
intercept = regr.intercept_
coef = regr.coef_

# estimator: regression coefficient
print(intercept, coef)


fig, ax0 = plt.subplots(nrows=1)
fig = plt.figure(figsize=(10, 8))

plt.rc('lines', linewidth=4)
plt.scatter(dx_test, dy_test, color='red')
plt.plot(dx_test, regr.predict(dx_test), color='blue')
plt.title('A simple OLS model')
plt.xlabel('input x')
plt.ylabel('predict y')
plt.show()


