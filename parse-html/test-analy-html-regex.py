# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from pandas import Series, DataFrame
# import os, sys
import re

file1 = '20040203073334.html'
source = open(file1, 'r').read()

gather = ['Enterprise Value']
print('**' * 39)
for each_data in gather:
    print('to find tags to be matched...')
    print(each_data in source)
    regex = re.compile(each_data + '.*?(\d{1,8}\.\d{1,8}B?M?)')
    search_result = re.search(regex, source)
    print(search_result.group(1))
