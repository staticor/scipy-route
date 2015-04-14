# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''
import os, sys, re
import time
from datetime import datetime
from pandas import DataFrame
from time import mktime
import matplotlib
from matplotlib import style
style.use('ggplot')
import matplotlib.pyplot as plt


if sys.platform.startswith('darwin'):
    path = '/Users/staticor/Downloads/intraQuarter/_KeyStats/'  # mac
else:
    path = 'D:/intraQuarter/_KeyStats/'  # Windows


def Key_Stats(gather= ["Total Debt/Equity (mrq)",
                       'Trailing P/E',
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
                       'Shares Short (prior ']):
    stock_list = [x[0] for x in os.walk(path)]
    df = DataFrame(columns=['Date',
                            'Unix',
                            'Ticker',
                            
                            #########
                            'DE Ratio',
                            'Trailing P/E',
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
                       'Shares Short (prior '
                            
                            
                            
                            ######
#                            'Price',
#                            'stock_p_change',
#                            'SP500',
#                            'sp500_p_change',
#                            'Difference',
                            'Status'])
                            
    sp500_df = DataFrame.from_csv('YAHOO-INDEX_GSPC.csv')
    
    ticker_list = []
    # define old price to calculate the percentage of change.
    
    
    for each_dir in stock_list[1:10]:
        each_file = os.listdir(each_dir)
        ticker = each_dir.split('/')[-1]
        ticker_list.append(ticker)
        
        start_stock_price = False
        start_sp500_price = False        
        
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                full_file_path = each_dir + '/' + file
                source = open(full_file_path, 'r').read()
                try:
                    value_list = []
                    for each_data in gather:
                        try:
                            regex = re.escape(each_data) + r'.*?(\d{1, 8}\.\d{1, 8}M?B?K?|N/A)%?</td>'
                            
                            
                        
                        except Exception as e:
                            value = "N/A"
                            value_list.append(value)
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
                            #print(str(e), ticker, file)

#                        time.sleep(15)
                    if not start_stock_price:
                        start_stock_price = stock_price

                    if not start_sp500_price:
                        start_sp500_price = sp500_value
                    change_percent_stock = ( (stock_price - start_stock_price)/start_stock_price) * 100
                    change_percent_sp500 = ( (sp500_value - start_sp500_price)/start_sp500_price) * 100


                    start_stock_price = stock_price
                    start_sp500_price = sp500_value

                    # status
                    difference = change_percent_stock - change_percent_sp500
                    if difference > 0 :
                        status = "outperform"
                    else:
                        status = "underperform"

                    df = df.append({'Date': date_stamp,
                    'Unix': unix_time,
                    'Ticker': ticker,
                    'DE Ratio': value,
                    'Price': stock_price,
                    'SP500': sp500_value,
                    'stock_p_change': change_percent_stock,
                    'sp500_p_change': change_percent_sp500,
                    'Difference': difference,
                    'Status': status
                    },

                    ignore_index=True)
                except Exception as e:
                    #print(str(e))
                    pass
                #time.sleep(3)

    for each_ticker in ticker_list:
        try:
            if each_ticker in ['a', 'aapl', 'abc', 'aci']:
                plot_df = df[(df['Ticker'] == each_ticker)]
                plot_df.set_index(['Date'])

                if plot_df['Status'][-1] == 'underperform':
                    color = 'r'
                else:
                    color = 'g'


                plt.plot(plot_df['Difference'], label=each_ticker, color=color)
                plt.legend()

        except:
            pass
    print(plot_df.head())
    print(ticker_list)
    plt.show()
    save = gather.replace(' ', '').replace('(', '').replace(')', '').replace('/', '') + ('.csv')
    print(save)
    df.to_csv(save)

#    df[['Price', 'SP500']].plot()
#    df[['% change of stock', '% change of SP500']].plot()

Key_Stats()
