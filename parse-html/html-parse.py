# -*- coding: utf-8 -*-
'''
Created on April , 2015
@author: stevey
'''

from bs4 import BeautifulSoup

target_file = 'test.html'
source = open(target_file, 'r').read()
# print(source)
soup = BeautifulSoup(source)


# get title
title = soup.find('title')
print(title)


# tbody
tbody = soup.find('tbody')
#print(tbody)


# get a td info.

td_title = soup.find("td", { "class": "title_block"})
print(td_title)


print("***********************************************************")

# REGEX

import re
content = '''
          <td class="cell_c">DKK</td>
          <td class="cell_c">100</td>
          <td class="cell">Danish Krone</td>

          <td class="cell_c">308.6106</td>
     </tr>

      <tr>
          <td class="cell_c">978</td>
          <td class="cell_c">EUR</td>
          <td class="cell_c">100</td>
          <td class="cell">EURO</td>
'''
gather = ['DKK', 'EUR', 'Danish Krone']
for each_data in gather:
    regex = re.escape(each_data) + r'.*?(\d{1,9}\.\d{0,9})</td>'
    search_result = re.search(regex, source)
    print(search_result)
    if search_result is not None:
        value = float(search_result.group(1))
        print('value found!:::', value)

