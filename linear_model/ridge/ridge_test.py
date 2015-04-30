
# ridge example of sklearn

from sklearn.linear_model import Ridge
import numpy as np

n_samples, n_features = 10, 1
np.random.seed(1428)

y = np.random.randn(n_samples)
X = np.random.randn(n_samples, n_features)

clf = Ridge(alpha=1.0)
F = clf.fit(X, y)
print(F.coef_, F.intercept_)

xm = np.mat(X)
w = np.mat(F.coef_)
print(xm.shape, w.shape)

# Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
#    normalize=False, solver='auto', tol=0.001)

y_predicted = w * xm.T + F.intercept_
'''
Ridge(alpha=1.0, copy_X=True, fit_intercept=True, max_iter=None,
   normalize=False, solver='auto', tol=0.001)
'''

from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=1)
ax.plot(X, y_predicted.T, color='blue')
ax.scatter(X, y, color='red')
plt.title('Example of ridge')
plt.show()