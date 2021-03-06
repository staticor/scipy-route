# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''
import os
import time
from datetime import datetime
from pandas import DataFrame
import sys

if sys.platform.startswith('darwin'):
    path = '/Users/staticor/Downloads/intraQuarter/_KeyStats/'  # mac
else:
    path = 'D:/intraQuarter/_KeyStats'  # Windows


def Key_Stats(gather="Total Debt/Equity (mrq)"):
    stock_list = [x[0] for x in os.walk(path)]
    df = DataFrame(columns=['Date', 'Unix', 'Ticker', 'DE Ratio'])
    for each_dir in stock_list[1:]:
        each_file = os.listdir(each_dir)
        ticker = each_dir.split('/')[-1]
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                full_file_path = each_dir + '/' + file
                print(full_file_path)
                source = open(full_file_path, 'r').read()
                try:
                    value = float(source.split(gather + ':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
                    df = df.append({'Date': date_stamp, 'Unix': unix_time, 'Ticker': ticker, 'DE Ratio': value}, ignore_index=True)
                except Exception as e:
                    pass
                #time.sleep(3)
    save =gather.replace(' ', '').replace('(', '').replace(')', '').replace('/', '') + ('.csv')
    df.to_csv(save)

Key_Stats()
