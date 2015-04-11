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
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import style
style.use('dark_background')
from time import mktime
import re

# define my data-source, depending on my platform
if sys.platform.startswith('darwin'):
    path = '/Users/staticor/Downloads/intraQuarter/_KeyStats/'  # mac
else:
    path = 'D:/intraQuarter/_KeyStats/'  # Windows


def Key_Stats(gather="Total Debt/Equity (mrq)"):
    stock_list = [x[0] for x in os.walk(path)]
    df = DataFrame(columns=['Date',
                            'Unix',
                            'Ticker',
                            'DE Ratio',
                            'Price',
                            'SP500',
                            '% change of stock',
                            '% change of SP500',
                            'Difference'
                            ])
    sp500_df = DataFrame.from_csv('YAHOO-INDEX_GSPC.csv')

    ticker_list = []

    # define old price to calculate the percentage of change.
    start_stock_price = False
    start_sp500_price = False
    for each_dir in stock_list[1:5]:
        each_file = os.listdir(each_dir)
        ticker = each_dir.split('/')[-1]
        ticker_list.append(ticker)
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

                    if not start_stock_price:
                        start_stock_price = stock_price
                    if not start_sp500_price:
                        start_sp500_price = sp500_value
                    change_percent_stock = ( (stock_price - start_stock_price)/start_stock_price) * 100
                    change_percent_sp500 = ( (sp500_value - start_sp500_price)/start_sp500_price) * 100


                    start_stock_price = stock_price
                    start_sp500_price = sp500_value
                    df = df.append({'Date': date_stamp,
                    'Unix': unix_time,
                    'Ticker': ticker,
                    'DE Ratio': value,
                    'Price': stock_price,
                    'SP500': sp500_value,
                    '% change of stock': change_percent_stock,
                    '% change of SP500': change_percent_sp500,
                    'Difference': change_percent_stock - change_percent_sp500
                    },

                    ignore_index=True)
                except Exception as e:
                    print(str(e))


    for each_ticker in ticker_list:
        try:
            plot_df = df[ (df['Ticker'] == each_ticker)]
            plot_df = plot_df.set_index(['Date'])
            plot_df['Difference'].plot(label=each_ticker)
            plt.legend()

        except:
            pass
                #time.sleep(3)
    save = gather.replace(' ', '').replace('(', '').replace(')', '').replace('/', '') + ('.csv')
    df.to_csv(save)

    df[['Price', 'SP500']].plot()
    df[['% change of stock', '% change of SP500']].plot()
    plt.show()
Key_Stats()
