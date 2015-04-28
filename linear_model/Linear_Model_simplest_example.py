
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')



input = [
    [1, 10],
    [2, 23],
    [3, 30],
    [4, 39],
    [5, 55],
    [6, 63.5]
]

data = np.array(input)
X = data[:, 0]
y = data[:, 1]
X.shape = (6, 1)
regr = linear_model.LinearRegression()
regr.fit(X, y)
y_predict = regr.predict(X)
print(regr.coef_, )
print(regr.intercept_)

# plot

plt.scatter(X, y, color='red')
plt.plot(X, regr.coef_[0] * X + regr.intercept_, linear
    linear'b--', linewidth=3)
plt.show()