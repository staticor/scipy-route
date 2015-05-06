# -*- coding: utf-8 -*-

# Created on May, 2015
# @author: stevey
#


import os, sys
import urllib2


url = 'http://qingg.im/donators.html'
url_local = 'qingge-donator.html'
request = urllib2.urlopen(url)
content = request.read()

# print(content)

from bs4 import BeautifulSoup
with open(url_local, 'r') as urlcontent:
    soup = BeautifulSoup(urlcontent.read())
    td1 = soup.find("td", {"class": "dt-column_1952"})
    print(td1)