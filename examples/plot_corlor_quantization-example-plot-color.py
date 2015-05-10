# -*- coding: utf-8 -*-

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
china = np.array(china, dtype=np.float64) / 255
print china.min(), china.max()
print china
# Load the image and transform to a 2D numpy array
w, h, d = original_shape = tuple(china.shape)
print w,h,d  # 427 * 640 * 3
assert d == 3
image_array = np.reshape(china, (w*h, d))
print(image_array[1])
print('Fitting model on a small sub-sample of the data')
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print('done in %0.3fs.' % (time()-t0))

# Get labels for all points
print('Predicting color indices on the full image(k-means')

t0 = time()
labels = kmeans.predict(image_array)
print('done in %0.3fs.' % (time()-t0))

codebook_random = shuffle(image_array, random_state=0)[:n_colors + 1]
print('codebook - random shape: ', codebook_random.shape)
# print(codebook_random[1])
print('Predicting color indices on the full image(random)')
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random, image_array, axis=0)
print('done in %0.3fs.' % (time() - t0))


def recreate_image(codebook, labels, w, h):
    '''Recreate the (compressed) image from the code book & labels '''
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_index = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_index]]
            label_index += 1
    return image

# Display all results , alongside original image

plt.figure(1)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Original china-jpg,  96 615 colors')
plt.imshow(china)

plt.figure(2)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64colors, K-means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))


plt.figure(3)
plt.clf()
ax = plt.axes([0, 0, 1, 1])
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()










