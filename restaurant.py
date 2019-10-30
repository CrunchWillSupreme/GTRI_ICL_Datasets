# -*- coding: utf-8 -*-
"""
GTRI Restaurant Inspection Analysis for GTRI Interview
@author: Will Han

"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt
import statsmodels.api as sm
from statsmodels.formula.api import ols
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 10)
pd.set_option('display.width', 150)

os.chdir(r'C:\Codes\GTRI_analysis')

# Load data
raw_data = pd.read_csv('https://raw.githubusercontent.com/CrunchWillSupreme/GTRI_ICL_Datasets/master/SF_Resturant_Scores/SF_Restaurant_Scores.csv')

# Check out the data
raw_data.head(5)
raw_data.columns
raw_data.describe()
raw_data.info()
raw_data.shape # (53583, 23)

# Check out the missing data
raw_data.isnull().sum()
raw_data.isnull().sum().sum()
raw_data.isnull().sum().plot.bar()
na_columns = raw_data.columns[raw_data.isnull().any()].to_list()
for column in na_columns:
    missing = raw_data[column].isnull().sum()
    percent_missing = missing/len(raw_data)*100
    print("Column '"+column+"' is missing "+str(percent_missing)+"%.")


# Are there specific inspection_types that do not have inspection_scores?
null_insp = raw_data['inspection_type'][raw_data['inspection_score'].isnull()]
print(null_insp.unique())

notnull_insp = raw_data['inspection_type'][raw_data['inspection_score'].notnull()]
print(notnull_insp.unique()) #'Routine - Unscheduled'

# is the inspection type that is present in the inspection_score = notnull ('Routine - Unscheduled'), also present in the inspection_score = Null? yes
notnull_insp.unique() in null_insp.unique()

sliced = raw_data[['business_id', 'business_name', 'inspection_score','inspection_date','inspection_type','violation_description','risk_category']].sort_values(by=['business_id','inspection_date'])
# Scores are based on violation risk category: Low Risk = -2, Moderate Risk = -4, High Risk = -7

raw_data.dtypes


def prep_data(raw_data):
    raw_data['inspection_date'] = pd.to_datetime(raw_data['inspection_date'])
    raw_data['violation_code'] = raw_data['violation_id'].str[-6:]
    raw_data['inspection_year'] = raw_data['inspection_date'].dt.year
    raw_data['inspection_month'] = raw_data['inspection_date'].dt.month
    raw_data = raw_data.rename(columns={"Police Districts": "Police_Districts","Supervisor Districts":"Supervisor_Districts", "Fire Prevention Districts":"Fire_Prevention_Districts","Zip Codes":"Zip_Codes","Analysis Neighborhoods":"Analysis_Neighborhoods"})
    data = raw_data.copy()
    return data

raw_data = prep_data(raw_data)

min(raw_data['inspection_date']),max(raw_data['inspection_date']) # inspections are from 2016-09-06 to 2019-11-28


# Get rows with inspection_type = 'Routine - Unscheduled' and inspection_score is null
null_score_routine = raw_data[(raw_data['inspection_type'] == 'Routine - Unscheduled') & (raw_data['inspection_score'].isnull())]
len(null_score_routine)/len(raw_data) 

# only 106 rows, ~0.2% of the data.  Let's remove these
data = raw_data[raw_data['inspection_score'].notnull()]

# take another look at the Routine - Unscheduled and inspection_score is null rows...
sort_null_scores = raw_data[raw_data['inspection_type'] == 'Routine - Unscheduled'].sort_values(by=['business_id','inspection_date','violation_description'])
# can calculate the inspection score, but will take some extra time to write code.

# see if violation_codes are unique to violation_descriptions
vio_codes = data[['violation_code','violation_description']].sort_values(by=['violation_code'])
vio_codes_list = vio_codes['violation_code'].unique()
for i in vio_codes_list:
    print(vio_codes['violation_description'][vio_codes['violation_code'] == i])

vio_dict = dict(zip(data['violation_code'],data['violation_description']))


# separate data by year
data_2016 = data[data['inspection_date'].dt.year == 2016]
data_2017 = data[data['inspection_date'].dt.year == 2017]
data_2018 = data[data['inspection_date'].dt.year == 2018]
data_2019 = data[data['inspection_date'].dt.year == 2019]

# are there any trends in average inspection_score across months? 
# bar plot averages of inspection_scores
bplot_data = data_2017['inspection_score'].groupby(data_2017['inspection_date'].dt.month).mean().to_frame(name='2017')
months = pd.Series(np.arange(0,13,1))
bplot_data['months'] = months
bplot_data['2018'] = data_2018['inspection_score'].groupby(data_2018['inspection_date'].dt.month).mean()
bplot_data['2016'] = data_2016['inspection_score'].groupby(data_2016['inspection_date'].dt.month).mean()
bplot_data['2019'] = data_2019['inspection_score'].groupby(data_2019['inspection_date'].dt.month).mean()
bplot_data = bplot_data[['months', '2016', '2017', '2018', '2019']]
bplot_data.plot(kind='area', stacked = False) # not very helpful
plt.figure; bplot_data.iloc[:,1:].plot()
plt.figure; bplot_data.iloc[:,1:].plot(kind = 'bar')
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
           ncol=2, mode="expand", borderaxespad=0.)

# non-visual averages or inspection_scores
data_2016['inspection_score'].groupby(data_2016['inspection_date'].dt.month).mean()
data_2017['inspection_score'].groupby(data_2017['inspection_date'].dt.month).mean()
data_2018['inspection_score'].groupby(data_2018['inspection_date'].dt.month).mean()
data_2019['inspection_score'].groupby(data_2019['inspection_date'].dt.month).mean()

# no obvious trends


# Violation code counts - are there some violations that are more common than others?
pd.crosstab(data_2017['violation_code'],data_2017['inspection_date'].dt.month).plot.bar()
# violation code vs inspection year - bar plot
pd.crosstab(data['violation_code'],data['inspection_year']).plot.bar()
# all violation codes - bar plot
data['violation_code'].value_counts().plot.bar()
ax = sns.countplot(x="violation_code", hue="risk_category", data=data)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
# violation codes vs inspection year - stacked bar plot
table=pd.crosstab(data['violation_code'], data['inspection_year'])
table.loc[:,:].plot.bar(stacked=True, figsize=(10,7))
#table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Violation Codes vs Inspection Year')
plt.xlabel('Inspection Codes')
plt.ylabel('Counts of Violation Codes')
plt.savefig('violation_codes_vs_inspection_year')


# view the counts of violations by months...is there a trend/seasonality?
data_2016['violation_code'].groupby(data_2016['inspection_date'].dt.month).count().plot.bar()
data_2017['violation_code'].groupby(data_2017['inspection_date'].dt.month).count().plot.bar()
data_2018['violation_code'].groupby(data_2018['inspection_date'].dt.month).count().plot.bar()
data_2019['violation_code'].groupby(data_2019['inspection_date'].dt.month).count().plot.bar()

pd.crosstab(data['inspection_year'], data['inspection_month']).plot.bar()
plt.title('Counts of Violations by Year and Month')
plt.legend(bbox_to_anchor=(1.01, 0.3, .3, .6), loc='lower left',
           ncol=1, mode="expand", borderaxespad=0.)
# drill down inspection_scores by police_district
data_2016['inspection_score'].groupby([data_2016['inspection_date'].dt.month, data_2016['Police_Districts']]).mean()
#pd.pivot_table(data_2017, index = [[data_2017['inspection_date'].dt.month,data_2017['Police Districts']]], values = [data_2017['inspection_score']], aggfunc=np.mean)
data_2017['inspection_score'].groupby([data_2017['inspection_date'].dt.month, data_2017['Police_Districts']]).mean()
data_2018['inspection_score'].groupby([data_2018['inspection_date'].dt.month, data_2018['Police_Districts']]).mean()
data_2019['inspection_score'].groupby([data_2019['inspection_date'].dt.month, data_2019['Police_Districts']]).mean()

# Distributions of inspection_score per month
dfs = [data_2016, data_2017, data_2018, data_2019]
i = 2016
for df in dfs:    
    g = sns.FacetGrid(df, col="inspection_month", col_wrap = 4)
    g.map(sns.distplot, "inspection_score")
#    g.xticks(np.arange(min("inspection_score"), max("inspection_score")+1, 5.0))
#    g.set(xticks=df['inspection_score'][::5])
    g.fig.suptitle(i)
    i+=1


# a look at inspection score averages by Police districts
df_2016 = pd.DataFrame(data_2016['inspection_score'].groupby([data_2016['inspection_date'].dt.month, data_2016['Police_Districts']]).mean())
df_2016.reset_index(inplace = True)
sns.barplot(x='inspection_date', y='inspection_score', hue = 'Police_Districts', data = df_2016)

df_2017 = pd.DataFrame(data_2017['inspection_score'].groupby([data_2017['inspection_date'].dt.month, data_2017['Police_Districts']]).mean())
df_2017.reset_index(inplace = True)
sns.barplot(x='inspection_date', y='inspection_score', hue = 'Police_Districts', data = df_2017)

df_2018 = pd.DataFrame(data_2018['inspection_score'].groupby([data_2018['inspection_date'].dt.month, data_2018['Police_Districts']]).mean())
df_2018.reset_index(inplace = True)
sns.barplot(x='inspection_date', y='inspection_score', hue = 'Police_Districts', data = df_2018)

df_2019 = pd.DataFrame(data_2019['inspection_score'].groupby([data_2019['inspection_date'].dt.month, data_2019['Police_Districts']]).mean())
df_2019.reset_index(inplace = True)
sns.barplot(x='inspection_date', y='inspection_score', hue = 'Police_Districts', data = df_2019)

# Run ANOVA to see if there are any statistically significant differences in inspection_scores between groups
results_PD = ols('inspection_score ~ C(Police_Districts)', data = data).fit()
results_PD.summary()
aov_table = sm.stats.anova_lm(results_PD, typ=2) 

results_SV = ols('inspection_score ~ C(Supervisor_Districts)', data = data).fit()
results_SV.summary()
aov_table = sm.stats.anova_lm(results_SV, typ=2) 

results_SV = ols('inspection_score ~ C(Fire_Prevention_Districts)', data = data).fit()
results_SV.summary()
aov_table = sm.stats.anova_lm(results_SV, typ=2)


# Time series of risk categories
ts_data = data[['inspection_date', 'risk_category']]
risk = pd.crosstab(ts_data['inspection_date'], ts_data['risk_category'])
risk.plot()

ts_data_2016 = data_2016[['inspection_date', 'risk_category']]
risk = pd.crosstab(ts_data_2016['inspection_date'], ts_data_2016['risk_category'])
risk.plot()

ts_data_2017 = data_2017[['inspection_date', 'risk_category']]
risk = pd.crosstab(ts_data_2017['inspection_date'], ts_data_2017['risk_category'])
risk.plot()

ts_data_2018 = data_2018[['inspection_date', 'risk_category']]
risk = pd.crosstab(ts_data_2018['inspection_date'], ts_data_2018['risk_category'])
risk.plot()

ts_data_2019 = data_2019[['inspection_date', 'risk_category']]
risk = pd.crosstab(ts_data_2019['inspection_date'], ts_data_2019['risk_category'])
risk.plot()



raw_data['business_state'].unique() # All in CA
raw_data['business_city'].unique() # All in San Francisco
len(raw_data['business_postal_code'].unique()) # 62 unique "zip codes"
raw_data.groupby('business_postal_code').count().head(5)
raw_data[raw_data['business_postal_code'] == '0']

na_data = raw_data[raw_data['Neighborhoods'].isnull()]
na_data.isnull().sum()
na_data['business_phone_number'].isnull().sum()
len(na_data)

neighborhood_zip = raw_data.groupby(['Analysis Neighborhoods', 'Zip Codes']).count()

# drop data where 'Analysis Neighborhoods' is null
data2 = raw_data[raw_data['Analysis Neighborhoods'].notnull()]
# same as below
subset = raw_data.dropna(subset = ['Analysis Neighborhoods'])
data2.isnull().sum()
data2.shape

# Drop all rows with NA in them
dropna_data = raw_data.dropna()
dropna_data.isnull().sum()
dropna_data.shape # (5275, 12) leaves us with less than 10% of original data

# Start by looking only at data with 'Inspection_score'
is_data = raw_data.dropna(subset = ['inspection_score'])
sns.distplot(is_data['inspection_score'])
is_data.reset_index(inplace = True)
is_data[['business_id','inspection_date']].sort_values(by=['business_id','inspection_date'])
is_data.isnull().sum()
is_data['inspection_score']
is_data['inspection_score'].groupby(is_data['Neighborhoods']).mean()
is_data['inspection_score'].groupby(is_data['Neighborhoods']).count()
is_data['inspection_score'].groupby(is_data['Police Districts']).mean()
is_data['inspection_score'].groupby(is_data['Police Districts']).count()
is_data['inspection_score'].groupby(is_data['Supervisor Districts']).mean()
is_data['inspection_score'].groupby(is_data['Supervisor Districts']).count()
is_data['inspection_score'].groupby(is_data['Fire Prevention Districts']).mean()
is_data['inspection_score'].groupby(is_data['Fire Prevention Districts']).count()
is_data['inspection_score'].groupby(is_data['Zip Codes']).mean()
is_data['inspection_score'].groupby(is_data['Zip Codes']).count()
is_data['inspection_score'].groupby(is_data['Analysis Neighborhoods']).mean()
is_data['inspection_score'].groupby(is_data['Analysis Neighborhoods']).count()




# creat bins for inspection_score
bins = [0, 25, 50, 60, 70, 80, 90, 100]
#labels = ['<25','<50','<60','<70','<80','<90','<100']
is_data['score_binned'] = pd.cut(is_data['inspection_score'], bins=bins)