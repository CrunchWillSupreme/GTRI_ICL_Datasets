# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 01:11:44 2019

@author: Will Han
"""

import requests
from lxml import html
from bs4 import BeautifulSoup
import json
from textblob import TextBlob
import pandas as pd

business = 'Hey Hey Gourmet'
location = '1 United Nations Plaza'
bus = business.replace(' ','+')
loc = location.replace(' ','+')
url = f'https://www.yelp.com/search?find_desc={bus}&find_loc={loc}+San+Francisco&ns=1'
url = r'https://www.yelp.com/biz/milk-and-cream-cereal-bar-new-york?osq=Ice+Cream'
response = requests.get(url)
#html_content = response.text()
soup = BeautifulSoup(response.content,'html5lib')
soup.prettify()
tags = soup.find_all('div',{'class':'lemon--div__373c0__1mboc u-space-b2 border-color--default__373c0__2oFDT'})
#tags = soup.find_all('span', {'class':'lemon--span__373c0__3997G'})

# Reviews
new_base=[]    
for i in tags:
    base = i.find('p', {'class':"lemon--p__373c0__3Qnnj text__373c0__2pB8f comment__373c0__3EKjH text-color--normal__373c0__K_MKN text-align--left__373c0__2pnx_"})
    if base != None:
        new_base.append(base)
    for j in new_base:
        review = j.find('span', {'class':'lemon--span__373c0__3997G'})
        print(review.text,'\n')
        
        
counts = soup.find_all('div',{'class':'lemon--div__373c0__1mboc u-space-b3 border-color--default__373c0__2oFDT'})
#tags = soup.find_all('span', {'class':'lemon--span__373c0__3997G'})

# trying to get the rating
new_countsbase=[]    
for i in counts:
    countsbase = i.find('span', {'class':"lemon--span__373c0__3997G display--inline__373c0__2q4au border-color--default__373c0__YEvMS"})
#    print(countsbase)
    if countsbase != None:
        new_countsbase.append(countsbase)
    for j in new_countsbase:
#        rating = j.find('div', {'class':'lemon--span__373c0__3997G'})
        print(j.text,'\n')
                   