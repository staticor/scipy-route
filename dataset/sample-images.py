# -*- coding: utf-8 -*-

# Created on May, 2015
# @author: stevey
#


from sklearn import datasets

# sample-images
# default 图像编码: uint8

data = datasets.load_sample_images()
print(len(data.images))
first_img_data = data.images[0]
second_img_data = data.images[1]
import matplotlib.pyplot as plt
# plt.imshow(first_img_data)
plt.imshow(second_img_data)
plt.show()
# plt.savejpg('sample-images1.jpg')
# plt.savejpg('sample-images2.jpg')