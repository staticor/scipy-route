# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import Series, DataFrame
import matplotlib
from matplotlib import style
from sklearn import svm, preprocessing
style.use('ggplot')
'''
Created on April , 2015
@author: stevey
@python love love love, you can make your best.!!
'''

FEATURES =  ['DE Ratio',
                       'Price/Sales',
                       'Price/Book',
                       'Profit Margin',
                       'Operating Margin',
                       'Return on Assets',
                       'Return on Equity',
                       'Revenue Per Share',
                       'Market Cap',
                       'Enterprise Value',
                       'Forward P/E',
                       'PEG Ratio',
                       'Enterprise Value/Revenue',
                       'Enterprise Value/EBITDA',
                       'Revenue',
                       'Gross Profit',
                       'EBITDA',
                       'Net Income Avl to Common ',
                       'Diluted EPS',
                       'Earnings Growth',
                       'Revenue Growth',
                       'Total Cash',
                       'Total Cash Per Share',
                       'Total Debt',
                       'Current Ratio',
                       'Book Value Per Share',
                       'Cash Flow',
                       'Beta',
                       'Held by Insiders',
                       'Held by Institutions',
                       'Shares Short (as of',
                       'Short Ratio',
                       'Short % of Float',
                       'Shares Short (prior ']
# Linear SVC learn

def build_data_set(features=FEATURES):
    data_df = pd.DataFrame.from_csv('key_stat.csv')
    data_df = data_df.fillna(0)
    data_df = data_df[:100]
    X = np.array(data_df[features].values ) #.tolist())
    y = (data_df['Status']
         .replace('underperform', 0)
         .replace('outperform', 1)
         .values.tolist())

    # 数据标准化
    X = preprocessing.scale(X)
    pdx = pd.DataFrame(X)
    #print(sum(pd.isnull(X)))
    # print(y)
    return  X, y


def Analysis():

    test_size = 500
    X, y = build_data_set()
    clf = svm.SVC(kernel='linear', C= 1.0)
    clf.fit(X, y)


    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min(X[:, 0]), max(X[:, 0]))
    yy = a * xx - clf.intercept_[0]/ w[1]
    h0 = plt.plot(xx, yy, 'k-', label='non weighted')

    plt.scatter(X[:, 0], X[:, 1])
    plt.show()

Analysis()
    #x = [[1,3], [1,5], [1,7]]




# 收获 怎样做数据标准化
# sklearn 中的 数据预处理模块.
# preprocessing, scale