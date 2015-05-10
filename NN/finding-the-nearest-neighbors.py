# -*- coding: utf-8 -*-

# Created on May, 2015
# @author: stevey
# http://scikit-learn.org/stable/modules/neighbors.html


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import datasets

from sklearn.neighbors import NearestNeighbors

X = np.array([[-1, -1], [-2, -1], [-3,-2], [1, 1], [2, 1], [3,2]])
nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)

distances, indices = nbrs.kneighbors(X)
print(indices)
print(distances)

print(nbrs.kneighbors_graph(X).toarray())

# KDTree and BallTree classes

from sklearn.neighbors import KDTree
kdt = KDTree(X, leaf_size=30, metric='euclidean')
print kdt.query(X, k=2, return_distance=False)







