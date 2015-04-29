# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''

import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import os, sys
from sklearn import datasets
style.use('ggplot')


# load data
diabetes = datasets.load_diabetes()
# select feature to analysis
diabetes_X = diabetes.data[:, np.newaxis]
dx_train = diabetes_X[:-20]
dx_test = diabetes_X[-20:]


