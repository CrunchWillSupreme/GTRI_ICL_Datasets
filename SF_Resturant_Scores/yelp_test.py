# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 00:16:21 2019

@author: Will Han
https://learn.co/lessons/python-api-intro-yelp

https://towardsdatascience.com/using-yelp-data-to-predict-restaurant-closure-8aafa4f72ad6
"""
import requests
client_id = 'lcLGwvwWrqvFz9wnKWfKQw'
api_key = 'rct8-9crlm9QFOda2A6P0spqInvN0UdWOuaB_AoUikjhmwPKRl5Z8XpJQNjatMhOKLxh1FSbqY28tlg5yqUe4hr8fg9FplqtIMaREuhG_elLtPCtmsYdXhb7_T_KXXYx'

term = 'Mexican'
location = 'Astoria NY'
SEARCH_LIMIT = 10

url = 'https://api.yelp.com/v3/businesses/search'

headers = {
        'Authorization': 'Bearer {}'.format(api_key),
    }

url_params = {
                'term': term.replace(' ', '+'),
                'location': location.replace(' ', '+'),
                'limit': SEARCH_LIMIT
            }
response = requests.get(url, headers=headers, params=url_params)
print(response)
print(type(response.text))
print(response.text[:1000])
# turn data into json
response.json()

import pandas as pd

df = pd.DataFrame.from_dict(response.json())

response.json().keys()
for key in response.json().keys():
    print(key)
    value = response.json()[key]
    print(type(value))
    print('\n\n')
    
response.json()['businesses'][:2]

response.json()['total']

response.json()['region']

df = pd.DataFrame.from_dict(response.json()['businesses'])
print(len(df))
print(df.columns)
df.head()


####################
## YELP API INTRO ##
####################
# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.

response = requests.get('https://api.yelp.com/v3/autocomplete?text=del&latitude=37.786882&longitude=-122.399972',headers = headers)
print(response.text)

response2 = requests.request('GET', 'https://api.yelp.com/v3/autocomplete?text=del&latitude=37.786882&longitude=-122.399972', headers=headers)

term = 'Buffalo Wild Wings'
location = 'Richmond VA'
SEARCH_LIMIT = 10

url = 'https://api.yelp.com/v3/businesses/search'

headers = {
        'Authorization': 'Bearer {}'.format(api_key),
    }

url_params = {
                'term': term.replace(' ', '+'),
                'location': location.replace(' ', '+'),
                'limit': SEARCH_LIMIT
            }
response = requests.get(url, headers=headers, params=url_params)
response.json().keys()
response.json()['businesses'].value_counts()

# to get business infor - only provides up to 3 photos and 3 reviews
GET https://api.yelp.com/v3/businesses/north-india-restaurant-san-francisco
# to get reviews of business
GET https://api.yelp.com/v3/businesses/north-india-restaurant-san-francisco/reviews
