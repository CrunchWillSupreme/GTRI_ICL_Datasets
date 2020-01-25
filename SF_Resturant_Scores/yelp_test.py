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

term = 'Popeyes'
location = 'Richmond VA'
SEARCH_LIMIT = 1
response_df = pd.DataFrame()
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
response.json()['businesses'][0]['alias']
#response.json()['total']
#response.json()['region']
df = pd.DataFrame.from_dict(response.json()['businesses'])
response_df = pd.concat([response_df, df])
df['latitude'] = response.json()['businesses'][0]['coordinates']['latitude']
df['alias'].iloc[0]
i= 0
while i < len(response.json()['businesses']):
    df['latitude'].iloc[i] = response.json()['businesses'][i]['coordinates']['latitude']
    i+=1

# to get business infor - only provides up to 3 photos and 3 reviews
GET https://api.yelp.com/v3/businesses/north-india-restaurant-san-francisco
# to get reviews of business
GET https://api.yelp.com/v3/businesses/north-india-restaurant-san-francisco/reviews


raw_data = pd.read_csv('https://raw.githubusercontent.com/CrunchWillSupreme/GTRI_ICL_Datasets/master/SF_Resturant_Scores/SF_Restaurant_Scores.csv')

def prep_data(raw_data):
    raw_data['inspection_date'] = pd.to_datetime(raw_data['inspection_date'])
    raw_data['violation_code'] = raw_data['violation_id'].str[-6:]
    raw_data['inspection_year'] = raw_data['inspection_date'].dt.year
    raw_data['inspection_month'] = raw_data['inspection_date'].dt.month
    raw_data = raw_data.rename(columns={"Police Districts": "Police_Districts","Supervisor Districts":"Supervisor_Districts", "Fire Prevention Districts":"Fire_Prevention_Districts","Zip Codes":"Zip_Codes","Analysis Neighborhoods":"Analysis_Neighborhoods"})
    data = raw_data.copy()
    return data

raw_data = prep_data(raw_data)
# only 106 rows, ~0.2% of the data.  Let's remove these
data = raw_data[raw_data['inspection_score'].notnull()]

API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  

this = 0
while this <2:
    for i,j in zip(data['business_name'], data['business_address']):
        term = i
        location = j
        SEARCH_LIMIT = 1
        url = 'https://api.yelp.com/v3/businesses/search'
        headers = {'Authorization': 'Bearer {}'.format(api_key),}
        url_params = {  'term': term.replace(' ', '+'),
                        'location': location.replace(' ', '+'),
                        'limit': SEARCH_LIMIT}
        response = requests.get(url, headers=headers, params=url_params)
        df = pd.DataFrame.from_dict(response.json()['businesses'])
        response_df = pd.concat([response_df, df])
        this +=1
        
