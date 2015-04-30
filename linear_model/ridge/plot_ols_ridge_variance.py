
# ridge \\ ols variance


# Code source: Gael Varoquaux
# Modified for documentation by Jaques Grobler
# License: BSD 3 clause


from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt

X_train = np.c_[0.5, 1.].T
y_train = [.5, 1]
X_test = np.c_[0, 2].T

np.random.seed(1428)

classifiers = dict(ols=linear_model.LinearRegression(),
    ridge=linear_model.Ridge(alpha=0.1))
print(classifiers)
fignum = 1


for name, clf in classifiers.items():
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    plt.title(name)
    ax = plt.axes([.12, .12, .8, .8])

    for _ in range(6):
        this_X = .1 * np.random.normal(size=(2,1)) + X_train
        clf.fit(this_X, y_train)

        ax.plot(X_test, clf.predict(X_test), color='.5')
        ax.scatter(this_X, y_train, s=3, c='.5', marker='o', zorder=10)

    clf.fit(this_X, y_train)
    ax.plot(X_test, clf.predict(X_test), linewidth=2, color='blue')
    ax.scatter(X_train, y_train, s=30, c='r', marker='+', zorder=10)



    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlabel('input x')
    ax.set_ylabel('output y')

    ax.set_xlim(0, 2)
    fignum += 1

plt.show()