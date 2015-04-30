
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
fig, ax0 = plt.subplots(nrows=1)
plt.scatter(X, y, color='red')
plt.rc('lines', linewidth=4)
plt.plot(X, regr.coef_[0] * X + regr.intercept_, 'k--', color='green')

score = regr.score(X, y)
rscore = 'R scores {0:<8}'.format(score)
fig.suptitle(rscore)

regr_equation = 'y = {w1}x  {intercept}'.format(w1=regr.coef_[0], intercept=regr.intercept_)
plt.xlabel('input x')
plt.ylabel('target y')
plt.title(regr_equation)

plt.show()