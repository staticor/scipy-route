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
import matplotlib.pyplot as plt
if sys.platform.startswith('darwin'):
    path = '/Users/staticor/Downloads/intraQuarter/_KeyStats/'  # mac
else:
    path = 'D:/intraQuarter/_KeyStats/'  # Windows


def Key_Stats(gather="Total Debt/Equity (mrq)"):
    stock_list = [x[0] for x in os.walk(path)]
    df = DataFrame(columns=['Date', 'Unix', 'Ticker', 'DE Ratio'])
    sp500_df = DataFrame.from_csv('YAHOO-INDEX_GSPC.csv')
    for each_dir in stock_list[1:30]:
        each_file = os.listdir(each_dir)
        ticker = each_dir.split('/')[-1]
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                full_file_path = each_dir + '/' + file
                #print(full_file_path)
                source = open(full_file_path, 'r').read()
                try:
                    value = float(source.split(gather + ':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])

                    try:
                        sp500_date = datetime.fromtimestamp(unix_time ).strftime('%Y-%m-%d')
                        row = sp500_df[ (sp500_df.index == sp500_date)]
                        sp500_value = float(row['Adjusted Close'])
                    except:
                        sp500_date = datetime.fromtimestamp(unix_time - 259200 ).strftime('%Y-%m-%d')
                        row = sp500_df[ (sp500_df.index == sp500_date)]
                        sp500_value = float(row['Adjusted Close'])
                    stock_price = float(source.split('</small><big><b>')[1].split('</b></big>')[0])
                    
                    df = df.append({'Date': date_stamp, 
                    'Unix': unix_time, 
                    'Ticker': ticker, 
                    'DE Ratio': value,
                    'Price': stock_price,
                    'SP500': sp500_value}, 
                    
                    ignore_index=True)
                except Exception as e:
                    pass
                #time.sleep(3)
    save =gather.replace(' ', '').replace('(', '').replace(')', '').replace('/', '') + ('.csv')
    df.to_csv(save)
    
    df[['Price', 'SP500']].plot()
    plt.show()
Key_Stats()
