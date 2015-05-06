# -*- coding: utf-8 -*-
'''
Created on May , 2015
@author: stevey
'''

import numpy as np
np.random.seed(1428)

from matplotlib import style
style.use('ggplot')

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces
from sklearn.utils.validation import check_random_state

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, RidgeCV



data = fetch_olivetti_faces()
targets = data.target
# print(targets)
# split data
print(data.images.shape) # 400 * 64 * 64
print(len(data.images))

data = data.images.reshape( (len(data.images), -1))
train = data[targets<30]
test = data[targets>=30]

n_faces= 5
rng= check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces,))
# print(face_ids)

test = test[face_ids, :]

n_pixels = data.shape[1]
print(n_pixels) # 4096
X_train = train[:, :np.ceil(0.5 * n_pixels)]
y_train = train[:, np.floor(0.5 * n_pixels):]
X_test = test[:, :np.ceil(0.5 * n_pixels)]
y_test = test[:, np.floor(0.5 * n_pixels):]


ESTIMATORS = {
    'extratree': ExtraTreesRegressor(n_estimators=10, max_features=32, random_state=0),
    'knn': KNeighborsRegressor(),
    'linear regression': LinearRegression(),
    'ridge': RidgeCV(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# raw face
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
fig = plt.figure(figsize=(20, 16) , dpi= 440)
for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1)
    else:
        sub = plt.subplot(n_faces, n_cols, i*n_cols+1, title='true faces')
    sub.axis('off')
    sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest')

    for j, est in enumerate(sorted(ESTIMATORS)):
        jigsaw_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2+j)
        else:
            sub = plt.subplot(n_faces, n_cols, i*n_cols+2+j, title= est)
        sub.axis('off')
        sub.imshow(jigsaw_face.reshape(image_shape), cmap=plt.cm.gray, interpolation='nearest')


plt.show()