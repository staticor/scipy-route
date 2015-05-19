import requests
from bs4 import BeautifulSoup
import csv
import pandas as pd

r= requests.get("http://www.walmart.com/search/?query=marvel&cat_id=4096_530598")
r.content
soup = BeautifulSoup(r.content)

g_data = soup.find_all("div", {"class" : "tile-conent"})
g_price = soup.find_all("div",{"class" : "item-price-container"})
g_star = soup.find_all("div",{"class" : "stars stars-small tile-row"})

from collections import defaultdict
data = defaultdict(list)

for product_title in g_data:
   a_product_tle = product_title.find_all("a","js-product-title")
   for text_product_tle in a_product_title :
      data['Product title'] = text_roduct_title.text

df = pd.DataFrame(data)
print df