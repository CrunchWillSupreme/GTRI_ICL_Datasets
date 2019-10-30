# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 22:50:49 2019

@author: Will Han
map references:
map tutorial - https://towardsdatascience.com/visualizing-data-at-the-zip-code-level-with-folium-d07ac983db20
geojson map - https://observablehq.com/@khxu/san-francisco-zip-codes-map
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', 150)

os.chdir(r'C:\Codes\GTRI_analysis')

# Load data
raw_data = pd.read_csv('https://raw.githubusercontent.com/CrunchWillSupreme/GTRI_ICL_Datasets/master/SF_Resturant_Scores/SF_Restaurant_Scores.csv')

def prep_data(raw_data):
    """changes inspection_date to datetime type, makes columns for violation_code,
    inspection_year, inspection_month, and removes spaces from column names"""
    raw_data['inspection_date'] = pd.to_datetime(raw_data['inspection_date'])
    raw_data['violation_code'] = raw_data['violation_id'].str[-6:]
    raw_data['inspection_year'] = raw_data['inspection_date'].dt.year
    raw_data['inspection_month'] = raw_data['inspection_date'].dt.month
    raw_data = raw_data.rename(columns={"Police Districts": "Police_Districts","Supervisor Districts":"Supervisor_Districts", "Fire Prevention Districts":"Fire_Prevention_Districts","Zip Codes":"Zip_Codes","Analysis Neighborhoods":"Analysis_Neighborhoods"})
    data = raw_data.copy()
    return data

data = prep_data(raw_data)
data = data[data['inspection_score'].notnull()]
data = data[data['business_longitude'].notnull()]
data = data[data['business_longitude'] != 0]

BBox = [data['business_longitude'].min(), data['business_longitude'].max(), data['business_latitude'].min(), data['business_latitude'].max()]
sf_map = plt.imread(r'C:\Codes\GTRI_analysis\resources\map4\san_fran_map.png')

fig, ax = plt.subplots(figsize = (8,7))
ax.scatter(data['business_longitude'], data['business_latitude'], zorder=1, alpha= 0.2, s=10, c='b', cmap=plt.get_cmap("autumn"))
ax.set_title('Plotting Spatial Data on San Francisco Map')
ax.set_xlim(BBox[0],BBox[1])
ax.set_ylim(BBox[2],BBox[3])
ax.imshow(sf_map, zorder=0, extent = BBox, aspect= 'equal')

# same but different
sf_map = plt.imread(r'C:\Codes\GTRI_analysis\resources\map4\san_fran_map.png')
data.plot(kind="scatter", x="business_longitude", y="business_latitude", alpha=0.4,
          c="inspection_score", cmap=plt.get_cmap("jet_r"))
plt.imshow(sf_map, zorder=0, extent = BBox, aspect= 'equal')
plt.show()

# Look at police_distrcts, fire_districts, zip_codes
# police districts
sf_map = plt.imread(r'C:\Codes\GTRI_analysis\resources\map4\san_fran_map.png')
data.plot(kind="scatter", x="business_longitude", y="business_latitude", alpha=0.2,
          c="Police_Districts", cmap=plt.get_cmap("Set1"))
plt.imshow(sf_map, zorder=0, extent = BBox, aspect= 'equal')
plt.show()

# same using seaborn
sf_map = plt.imread(r'C:\Codes\GTRI_analysis\resources\map4\san_fran_map.png')
ax = sns.scatterplot(x="business_longitude", y="business_latitude", hue="inspection_score", size = "Police_Districts", data=data, palette = 'jet_r') #'RdBu'
#data.plot(kind="scatter", x="business_longitude", y="business_latitude", alpha=0.4,
#          c="inspection_score", cmap=plt.get_cmap("jet_r"))
plt.imshow(sf_map, zorder=0, extent = BBox, aspect= 'equal')
plt.show()


#sf_map = plt.imread(r'C:\Codes\GTRI_analysis\resources\map4\san_fran_map.png')
##ax = sns.scatterplot(x="business_longitude", y="business_latitude", hue="inspection_score", data=data, palette = 'jet_r') #'RdBu'
#ax2 = sns.scatterplot(x="business_longitude", y="business_latitude", hue="Police_Districts", data=data, edgecolor=sns.color_palette("jet_r", 10), alpha=0.1) #'RdBu'
##data.plot(kind="scatter", x="business_longitude", y="business_latitude", alpha=0.4,
##          c="inspection_score", cmap=plt.get_cmap("jet_r"))
#plt.imshow(sf_map, zorder=0, extent = BBox, aspect= 'equal')
#plt.show()
