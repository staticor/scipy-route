# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import os, sys

names = ['Age', 'Education',   'Income' , 'Status'  , 'Purchase?']
data = pd.read_csv('dt.txt', header=None, names = names, sep=r"\s{2,10}")


output = 'out.csv'
data['new_age'] = np.where(data['Age'] == '36-55', 3,
        np.where(data['Age'] == 'over55', 4,
            np.where(data['Age'] == 'belo18', 1,
                np.where(data['Age'] == '18-35', 2, None)
                )
            )
        )
data['new_Edu'] = np.where(data['Education'].contains('master'), 3, )
print(data['Education'].unique())
# data.to_csv(output)

print(data.head())

# http://www.onlamp.com/lpt/a/6464
# 自己造轮子

def create_decision_tree(data, attributes, target_attr, fitness_func):
    '''
    Return a new decision tree based on the examples given.
    '''
    data = data[:]
    vals = [record[target_attr] for record in data]
    default = None
    #majority_value(data, target_attr) raw



    # If the dataset is empty or the attributes list is empty, return the
    # default value. When checking the attributes list for emptiness, we
    # need to subtract 1 to account for the target attribute.
    # 如果数据集为空, 或 目标变量列表为空
    if not data or (len(attributes) - 1) <= 0:
        return default

    # If all the records in the dataset have the same classification,
    # return that classification. 如果所有记录都是相同类别 就只能返回这一类别
    if vals.count(vals[0]) == len(vals):
        return vals[0]


    # 选择最优的分类变量
    best = choose_attribute(data, attributes, target_attr, fitness_func)

    # 用找到的最优变量 建立一个树
    tree = {best: {}}

    for val in get_values(data, best):
        subtree = create_decision_tree(
                get_examples(data, best, val),
                [attr for attr in attributes if attr != best],
                target_attr, fitness_func
            )
        tree[best][val] = subtree

    return tree


# from sklearn import tree

# classifier = tree.DecisionTreeClassifier()

# training_data_x = data[:15]['Age', 'Education', 'Income', 'Marital', 'Status']
# training_data_y = data[:15]
# forecast_data = data[15:]

