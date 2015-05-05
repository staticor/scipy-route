# -*- coding: utf-8 -*-
"""
Created on Tue May 05 10:03:20 2015

@author: STEVE
"""

import numpy as np
np.random.seed(1428)

import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import seaborn as sns
sns.set(palette='Set2')


# create input data, based a sine curve
# add random effect


def sine_wave(n_x, obs_err_sd=1.5, tp_err_sd=.3):
    x = np.linspace(0, (n_x - 1) / 2, n_x)
    y = np.sin(x) + np.random.normal(0, obs_err_sd) + np.random.normal(0, tp_err_sd, n_x)
    return y


sines = np.array([sine_wave(31) for _ in range(20)])
sns.tsplot(sines);




# # random walk

# def random_walk(n, start=0, p_inc=0.2):
#     return start + np.cumsum(np.random.uniform(size=n) < p_inc)


# starts = np.random.choice(range(4), 10)
# probs = [.1, .3, .5]
# walks = np.dstack([random_walk(15, s, p) for s in starts] for p in probs)

# sns.tsplot(walks)


# tutorial of git in liaoxuefeng.com  tsplot
# date: 05May2015
list_ = [6357, 1007, 874, 827, 1530, 1785, 1021, 1051, 696,646, 617, 505,\
821, 1147, 766, 476, 861, 551, 445, 336, 268, 462, 182, 202, 153, 819, 195, 204, 151, 552, 157]
plt.figure()
sum_ = sum(list_)
list_ = [i / float(sum_) for i in list_]
plt.scatter(list(range(1, len(list_)+1)), list_, color='g', marker='o')
plt.plot(list(range(1, len(list_)+1)), list_, 'r--', linewidth=2)
plt.xlim([0, 40])