# code: plot_iris.py


import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from matplotlib import style
style.use('dark_background')

# import some data.
iris = datasets.load_iris()
# print(iris.data.shape) # 150*4, is input observations' shape.
X = iris.data[:, :2] # Only consider the last 2 input variable.
y = iris.target
y_color_mapping = {0: 'r', 1: 'yellow', 2: 'blue'}
h = 0.02 # step size of mesh, for the unit of X_Y plane
C = 1.0  # Regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)


# create mesh to plot
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


# title for plots

titles = ['SVC with linear kernel',
            'LinearSVC (linear kernel)',
            'SVC with RBF kernel',
            'SVC with polynomial (degree 3) kernel']

for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot decision boundary. For that,
    # assign a color to each point in the mesh [x_min, x_max]
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # put result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, cmap=plt.cm.Paired, alpha=.8)
    plt.scatter(X[:, 0], X[:, 1], c=[y_color_mapping[i] for i in y], cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()





