# -*- coding: utf-8 -*-

# Created on May, 2015
# @author: stevey
# Face completion with a multi-output estimators


import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state
from sklearn.linear_model import LinearRegression, RidgeCV

# Load the faces datasets
#

data = fetch_olivetti_faces()
targets = data.target
data = data.images.reshape((len(data.images), -1))
# Split dataset -- train  target   }}} test: >=30
train = data[targets < 30]
test = data[targets >= 30]

print('the test-shape:',test.shape) # 100, 4096
# Test on a subset of people
n_faces =5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
print face_ids
test = test[face_ids, :]
n_pixels = data.shape[1]
X_train = train[:, : np.ceil(0.5*n_pixels)] # 上半张脸
y_train = train[:, np.floor(0.5* n_pixels):] # 下半张脸

X_test = test[:, :np.ceil(0.5*n_pixels)]
y_test = test[:, np.floor(0.5*n_pixels):]

# Fit estimators

ESTIMATORS = {
    "Extra trees": ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
    "K-nn": KNeighborsRegressor(),
    "Linear regression": LinearRegression(),
    "Ridge": RidgeCV(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# plot the complete faces
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2 * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack( (X_test[i], y_test[i]) )
    if i :
        sub = plt.subplot(n_faces, n_cols, i * n_cols+1)
    else:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1, title="true faces")

    sub.axis('off')
    sub.imshow(true_face.reshape(image_shape),
        cmap=plt.cm.gray,
        interpolation='nearest'
        )

    for j, est in enumerate( sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i :
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2+j, title=est)

        sub.axis('off')
        sub.imshow(completed_face.reshape(image_shape), cmap=plt.cm.gray,
            interpolation='nearest')

plt.show()

print('done')




