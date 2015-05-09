# -*- coding: utf-8 -*-

# Created on May, 2015
# @author: stevey
#


from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
china = load_sample_image('china.jpg')
print(china.dtype)
print(china.shape) # 427 640 3

# print(china[1, 2, :])
plt.imshow(china)
plt.title('Original image 96,615 colors')
plt.show() 2