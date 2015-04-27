
'''
Linear Model:
Part1: OLS Example
'''

from sklearn import linear_model, datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
clf = linear_model.LinearRegression()
print(clf.fit([[0,0], [1,0.9], [2,2.1]], [0,1,2]))
print(clf.coef_, clf.intercept_)

diabetes = datasets.load_diabetes()
diabets_x_data = diabetes.data[:, np.newaxis]
diabets_x_data = diabets_x_data[:, :, 2]

diabets_X_train = diabets_x_data[:-20]
diabets_X_test = diabets_x_data[-20:]
diabets_y_train = diabetes.target[:-20]
diabets_y_test = diabetes.target[-20:]
# create linear regression object
regr = linear_model.LinearRegression()
regr.fit(diabets_X_train, diabets_y_train)

# Output model
## The coefficients and MSE
print('Coefficients: \n', regr.coef_)
mse = np.mean((regr.predict(diabets_X_test) - diabets_y_test) ** 2)
print('MSE : ', mse)
## Explained Variance score(goodness of fitting)
print('Variance Score', regr.score(diabets_X_test, diabets_y_test))

## Plot outputs
plt.scatter(diabets_X_test, diabets_y_test, color='k')
plt.plot(diabets_X_test, regr.predict(diabets_X_test), color='blue', linewidth=3)
plt.xticks()
plt.yticks()
plt.xlabel('Input X')
plt.ylabel('Output Y')
plt.show() # http://ww3.sinaimg.cn/large/5810d07bjw1erkhc8po4fj20y60poabh.jpg



# Ordinary Least Squares Complexity
'''
This method computes the least squares solution using a singular
value decomposition of X.
if X is a matrix of size(n, p). method has a cost of O(np^2), assuming that n>=p

'''

## OLS example






## Ridge Regression
## addresses OLS by imposing a penalty on the size of coefficients.
## The ridge coefficients minimize a penalized residual sum of squares.



