# -*- coding: utf-8 -*-

'''
Created on May , 2015
@author: stevey
source idea: numpy xercises:
reference: http://www.labri.fr/perso/nrougier/teaching/numpy.100/index.html
'''

import numpy as np
np.random.seed(1428)
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import os, sys
from sklearn import datasets
style.use('ggplot')


def oye():
    print('-'*79)

oye()

########################################################################

# Lets simplify the quick-part
# import   import numpy as np
# check numpy version and configuration
print np.__version__   # 1.9.2 env: my PC @yixin
# print np.__config__.show()   # feel: less to use

########################################################################

# data(vector) initial

Z = np.zeros(9); print Z;
Z[4] = 999; print Z;
print(np.arange(3, 13, 2)) # [ 3  5  7  9 11]
## matrix
print Z.reshape(3, 3)
## identity matrix (eye matrix)  ones matrix.
print np.eye(3); print(np.ones((3,2)))

## diagonal matrix
print(np.diag([1,2,3]))
## random values matrix
print(np.random.random((1,2,3)))

# linspace like matlab
z = np.linspace(0, 1, 12, endpoint=True); print(z)
########################################################################
# min max mean
np.random.seed(1428)
z = np.random.random((1, 10))
print(z)
print z.min(), z.max(), z.mean()

# Normalize method1
z_nomal = (z-z.min())/(z.max() - z.min()); print(z_nomal)
# vector (matrix) dot product
z = np.dot(np.ones((5, 3)), np.ones((3, 2)) ); print(z)
########################################################################

## sort
z = np.random.random(5);
print(z) # before sorting
print(z.sort()) ### it return none
print(z)


########################################################################
# Make an array immutable (read-only)

Z = np.zeros(10); Z.flags.writeable = False
try:
    Z[0] = 1
except Exception as e:
    print(str(e) + ':::::::::Exception happend..')



########################################################################
# argmax argmin
print(z)
print(z.argmax(), z.argmin())






# #. Find indices of non-zero elements from [1,2,0,0,4,0]
aset = [1,2,0,0,4,0]
# print(aset.arg)