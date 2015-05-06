# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier
from pandas import Series, DataFrame

'''
Created on April , 2015
@author: stevey
@python love love love, you can make your best.!!
'''

# dummy data::
df = DataFrame({'c1': [0, 1, 2, 3], 'c2': [3, 4, 5, 6],
               'dv': [0, 1, 0, 1] })

# create tree

dt = DecisionTreeClassifier(max_depth=5, min_samples_split=1)
dt.fit(df[:, :2], df.dv)

from StringIO import StringIO
