
import numpy as np
from sklearn import linear_model, datasets
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# ols


# diabetes datasets
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]
print(diabetes_X_temp)



# splitting data, training and testing (left 20)

diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]
diabetes_y_train = diabetes.target[:-20]
diabetes_y_test = diabetes.target[-20:]
diabetes_y_train = diabetes.target[:-20]

# build model

regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train, diabetes_y_train)
print('Coef_\n', regr.coef_)

print('Residual sum of squares- MSE %.2f' % np.mean( (regr.predict(diabetes_X_test) - diabetes_y_test) ** 2 ))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(diabetes_X_test, diabetes_y_test))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test,  color='black')
plt.plot(diabetes_X_test, regr.predict(diabetes_X_test), color='blue',
         linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()


