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
from time import mktime
import matplotlib
from matplotlib import style
style.use('dark_background')
import matplotlib.pyplot as plt
import re 


if sys.platform.startswith('darwin'):
    path = '/Users/staticor/Downloads/intraQuarter/_KeyStats/'  # mac
else:
    path = 'D:/intraQuarter/_KeyStats/'  # Windows


def Key_Stats(gather="Total Debt/Equity (mrq)"):
    stock_list = [x[0] for x in os.walk(path)]
    df = DataFrame(columns=['Date', 'Unix', 'Ticker', 'DE Ratio'
    ,'Price', 'stock_p_change', 'SP500', 'sp500_p_change', 'Differnce'])
    sp500_df = DataFrame.from_csv('YAHOO-INDEX_GSPC.csv')
    ticker_list = []
    # define old price to calculate the percentage of change.
    start_stock_price = False
    start_sp500_price = False
    for each_dir in stock_list[1:100]:
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
                    try:
                        value = float(source.split(gather + ':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
                    except Exception as e:
                        value = float(source.split(gather + ':</td>\n<td class="yfnc_tabledata1">')[1].split('</td>')[0])
                        #print(str(e), ticker, file)
                        
                    try:
                        sp500_date = datetime.fromtimestamp(unix_time ).strftime('%Y-%m-%d')
                        row = sp500_df[ (sp500_df.index == sp500_date)]
                        sp500_value = float(row['Adjusted Close'])
                    except:
                        sp500_date = datetime.fromtimestamp(unix_time - 259200 ).strftime('%Y-%m-%d')
                        row = sp500_df[ (sp500_df.index == sp500_date)]
                        sp500_value = float(row['Adjusted Close'])
                    try:
                        stock_price = float(source.split('</small><big><b>')[1].split('</b></big>')[0])
                    except Exception as e:
                        # some convert error fixed
                        try:
                            delimitor = '</small><big><b>'
                            stock_price = source.split(delimitor)[1].split('</b></big>')[0]
                            stock_price = re.search(r'(\d{1, 8}\.\d{1, 8})', stock_price)
                            stock_price = float(stock_price.group(1))
                            
#                           
                        except Exception as e:
                            delimitor = '<span class="time_rtq_ticker">'
                            stock_price = source.split(delimitor)[1].split('</b></big>')[0]
                            stock_price = re.search(r'(\d{1, 8}\.\d{1, 8})', stock_price)
                            stock_price = float(stock_price.group(1))
                            print(str(e), ticker, file)
                            
#                        time.sleep(15)
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
                    'stock_p_change': change_percent_stock,
                    'sp500_p_change': change_percent_sp500,
                    'Difference': change_percent_stock - change_percent_sp500
                    },

                    ignore_index=True)
                except Exception as e:
                    print(str(e))
                #time.sleep(3)
    
    for each_ticker in ticker_list:
        try:
            plot_df = df[ (df['Ticker'] == each_ticker)]
            plot_df.set_index(['Date'])
            plot_df['Difference'].plot(label=each_ticker)
            plt.legend()
        except:
            pass
    save = gather.replace(' ', '').replace('(', '').replace(')', '').replace('/', '') + ('.csv')
    df.to_csv(save)

#    df[['Price', 'SP500']].plot()
#    df[['% change of stock', '% change of SP500']].plot()
    plt.show()
Key_Stats()
