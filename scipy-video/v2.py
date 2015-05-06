# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''
import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import sys

if sys.platform.startswith('darwin'):
    path = '/Users/staticor/Downloads/intraQuarter/_KeyStats/' # mac
else:
    path = 'D:/intraQuarter/_KeyStats'# Windows

def Key_Stats(gather="Total Debt/Equity (mrq)"):
    stock_list = [ x[0] for x in os.walk(path)]
    #print(stock_list)
    for each_dir in stock_list[1:]:
        each_file = os.listdir(each_dir)
        ticker = each_dir.split('/')[-1]
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                #print(date_stamp, unix_time)
                full_file_path = each_dir + '/' + file
                print(full_file_path)
                source = open(full_file_path, 'r').read()
                #print(source)
                try:
                    value = float(source.split(gather+':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
                    print(ticker+':' ,  value)
                except Exception as e:
                    pass
                time.sleep(3)



Key_Stats()

