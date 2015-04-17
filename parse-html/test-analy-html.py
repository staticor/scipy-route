# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''

import re
from bs4 import BeautifulSoup


file1 = '20040203073334.html'
source = open(file1, 'r').read()

gather = ['Enterprise Value']
print('**' * 39)

soup = BeautifulSoup(source)
table = soup.find("table", { "class": "yfnc_datamodoutline1"})

# print(table)
for row in table.findAll("tr"):
    cells = row.findAll("td")
    cell = row.find('td', {'class': 'yfnc_tablehead1'} )
    print(cell)
    print('*' * 20 + 'row over' + '*' * 20)

# for each_data in gather:
#     regex = re.compile(each_data+ r'.*?(\d{1, 8}\.?\d{1, 8})</td>', re.DOTALL)
#     try:
#         value = re.match(regex, source)#.group(1)
#         dirty = re.compile(r'yfnc_tabledata1.*?>(\d{1,9}\.\d{1, 9})</td>' )
#         c = source.split(each_data)[1]
#         print(re.search(dirty, c))

#     except Exception as e:
#         print(str(e), 'not found')

# from HTMLParser import HTMLParser
# hp = HTMLParser()
# print(hp.feed(source))


# import urllib2
# from bs4 import BeautifulSoup

# contenturl = "http://www.bank.gov.ua/control/en/curmetal/detail/currency?period=daily"
# soup = BeautifulSoup(urllib2.urlopen(contenturl).read())

# table = soup.find('div', attrs={'class': 'content'})
# rows = table.findAll('tr')
# print(rows)

# for tr in rows:
#     cols = tr.findAll('td')
#     for td in cols:
#         text = td.find(text=True) + ';'
#         print text,
#     print