from __future__ import print_function
# Lsso

from sklearn import linear_model
clf = linear_model.Lasso(alpha=0.1)
print(clf.fit([[0, 0], [1, 1]], [0, 1]))

'''output

Lasso(alpha=0.1, copy_X=True, fit_intercept=True, max_iter=1000,
   normalize=False, positive=False, precompute='auto', tol=0.0001,
   warm_start=False)

'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from sklearn import cross_validation, datasets, linear_model

diabetes = datasets.load_diabetes()
X = diabetes.data[:150]
y = diabetes.target[: 150]

lasso = linear_model.Lasso()
alphas = np.logspace(-4, -.5, 30)
# print(alpha)
scores = list()
scores_std = list()

for alpha in alphas:
    lasso.alpha = alpha
    this_scores = cross_validation.cross_val_score(lasso, X, y, n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))


plt.figure(figsize=(4, 3))
plt.semilogx(alphas, scores)

# plot error lines showing +/- std. errors of the scores .

plt.semilogx(alphas, np.array(scores), color='red')
plt.semilogx(alphas, np.array(scores) + np.array(scores_std)/np.sqrt(len(X)), 'b--')
plt.semilogx(alphas, np.array(scores) - np.array(scores_std)/np.sqrt(len(X)), 'b--')
plt.ylabel('CV score')
plt.xlabel(('alpha'))
plt.axhline(np.max(scores), linestyle='--', color='.5')


plt.show()


# how much can you trust the selection of alpha
## use external cross-validation to see how much the automatically obtained alpha differ across different cross-validation folds.

lasso_cv = linear_model.LassoCV(alphas=alphas)
k_fold = cross_validation.KFold(len(X), 3)
print("Answer to the bonus question:",
      "how much can you trust the selection of alpha?")
print()
print("Alpha parameters maximising the generalization score on different")
print("subsets of the data:")
for k, (train, test) in enumerate(k_fold):
    lasso_cv.fit(X[train], y[train])
    print("[fold {0}] alpha: {1:.5f}, score: {2:.5f}".
          format(k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])))
print()
print("Answer: Not very much since we obtained different alphas for different")
print("subsets of the data and moreover, the scores for these alphas differ")
print("quite substantially.")
