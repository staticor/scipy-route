
# Created on May, 2015
# Staticor copy

# Authors: Robert Layton <robertlayton@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#
# License: BSD 3 clause


# Instance of Color quantization -- cluster plot using kmeans
# http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#example-cluster-plot-color-quantization-py

from time import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import datasets


from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

# Specified the number of colors which were compresses
n_colors = 64

# Load the photo -- china summer palace
china = load_sample_image('china.jpg')


# convert: to float -- instead of default 8 bit integer
# Dividing by 255 so that plt.imshow behaves works well on float data
# need to be in the range [0 - 1]
china = np.array(china, dtype=np.float64)
print china.min(), china.max()
print china.shape

# raw_data = []
# for i in range(427):
#     for j in range(640):
#         x = str(china[i][j][0])  + str(china[i][j][1]) + str(china[i][j][2])
#         print(x)
#         if x not in raw_data:
#              raw_data.append(x)

# print len(raw_data)

