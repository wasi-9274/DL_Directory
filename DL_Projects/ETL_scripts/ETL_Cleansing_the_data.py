import pandas as pd
import numpy as np

#
# df_indicator = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/population_data.csv',
#                            skiprows=4)
# df_indicator.drop(["Unnamed: 62"], axis=1, inplace=True)
#
# df_projects = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/projects_data.csv',
#                           dtype=str)
# df_projects.drop(['Unnamed: 56'], axis=1, inplace=True)
#
# # print(df_indicator[['Country Name', 'Country Code']].drop_duplicates())
#
# # print(df_projects['countryname'].unique())
#
# df_projects['Official Country Name'] = df_projects['countryname'].str.split(';').str.get(0)
# print(df_projects)


# from pycountry import countries
#
# # print(countries.get(name="Spain"))
# #
# # print(countries.lookup('Kingdom of Spain'))
#
# from collections import defaultdict
#
# country_not_found = []
#
# project_country_abbrev_dict = defaultdict(str)
#
# for country in df_projects['Official Country Name'].drop_duplicates().sort_values():
#     try:
#         project_country_abbrev_dict[country] = countries.lookup(country).alpha_3
#     except:
#         # print(country, 'not found')
#         country_not_found.append(country)
#
# country_not_found_mapping = {'Co-operative Republic of Guyana': 'GUY',
#                              'Commonwealth of Australia':'AUS',
#                              'Democratic Republic of Sao Tome and Prin':'STP',
#                              'Democratic Republic of the Congo':'COD',
#                              'Democratic Socialist Republic of Sri Lan':'LKA',
#                              'East Asia and Pacific':'EAS',
#                              'Europe and Central Asia': 'ECS',
#                              'Islamic  Republic of Afghanistan':'AFG',
#                              'Latin America':'LCN',
#                              'Caribbean':'LCN',
#                              'Macedonia':'MKD',
#                              'Middle East and North Africa':'MEA',
#                              'Oriental Republic of Uruguay':'URY',
#                              'Republic of Congo':'COG',
#                              "Republic of Cote d'Ivoire":'CIV',
#                              'Republic of Korea':'KOR',
#                              'Republic of Niger':'NER',
#                              'Republic of Kosovo':'XKX',
#                              'Republic of Rwanda':'RWA',
#                              'Republic of The Gambia':'GMB',
#                              'Republic of Togo':'TGO',
#                              'Republic of the Union of Myanmar':'MMR',
#                              'Republica Bolivariana de Venezuela':'VEN',
#                              'Sint Maarten':'SXM',
#                              "Socialist People's Libyan Arab Jamahiriy":'LBY',
#                              'Socialist Republic of Vietnam':'VNM',
#                              'Somali Democratic Republic':'SOM',
#                              'South Asia':'SAS',
#                              'St. Kitts and Nevis':'KNA',
#                              'St. Lucia':'LCA',
#                              'St. Vincent and the Grenadines':'VCT',
#                              'State of Eritrea':'ERI',
#                              'The Independent State of Papua New Guine':'PNG',
#                              'West Bank and Gaza':'PSE',
#                              'World':'WLD'}
#
#
# # print(project_country_abbrev_dict)
#
# project_country_abbrev_dict.update(country_not_found_mapping)
#
# df_projects['Country Code'] = df_projects['Official Country Name'].apply(lambda x: project_country_abbrev_dict[x])
#
# # print(df_projects.head())
# print(df_projects[df_projects['Country Code'] == ''])


# Data types Exercise:

# df_indicator = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/population_data.csv',
#                            skiprows=4)
# df_indicator.drop(['Unnamed: 62'], axis=1, inplace=True)
#
# df_projects = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/projects_data.csv',
#                           dtype=str)
# df_projects.drop(['Unnamed: 56'], axis=1, inplace=True)

# print(df_indicator.dtypes)

# keepcol = []
# for i in range(1960, 2018, 1):
#     keepcol.append(str(i))
#
# df_nafta = df_indicator[(df_indicator['Country Name'] == 'Canada') | (df_indicator['Country Name'] == 'United States')
#                         | (df_indicator['Country Name'] == 'Mexico')]
# print(df_nafta.sum(axis=0)[keepcol])
# print()

# print(df_projects.dtypes)
#
# print(df_projects[['totalamt', 'lendprojectcost']].head())
#
# print(df_projects['totalamt'].sum())
# print()
#
# #  to convert the str to numeric we use to_numeric
#
# df_projects['totalamt'] = pd.to_numeric(df_projects['totalamt'].str.replace(',', ''))
# print(df_projects['totalamt'].sum())

# PARSING THE DATES

# parsed_date = pd.to_datetime('January 1st 2017')
# print(parsed_date)
#
# print(parsed_date.month)
# print(parsed_date.year)
# print(parsed_date.second)
#
# parsed_date = pd.to_datetime('5/3/2017 5:30')
# print(parsed_date.month)
#
# parsed_date = pd.to_datetime('3/5/2017 5:30', format='%d/%m/%Y %H:%M')
# print(parsed_date.month)
#
# parsed_date = pd.to_datetime('5/3/2017 5:30', format='%m/%d/%Y %H:%M')
# print(parsed_date.month)

# he formatting abbreviations are actually part of the python standard. You can see examples at-
# http://strftime.org/

df_projects = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/projects_data.csv',
                          dtype=str)
df_projects.drop(['Unnamed: 56'], axis=1, inplace=True)
# print(df_projects.columns)

# print(df_projects.head(15)[['boardapprovaldate', 'board_approval_month', 'closingdate']])

df_projects['boardapprovaldate'] = pd.to_datetime(df_projects['boardapprovaldate'])
df_projects['closingdate'] = pd.to_datetime(df_projects['closingdate'])

# print(df_projects['boardapprovaldate'].dt.second)
# print(df_projects['boardapprovaldate'].dt.month)
# print(df_projects['boardapprovaldate'].dt.weekday)

df_projects['approvalyear'] = df_projects['boardapprovaldate'].dt.year
df_projects['approvalday'] = df_projects['boardapprovaldate'].dt.day
df_projects['approvalweekday'] = df_projects['boardapprovaldate'].dt.weekday
df_projects['closingyear'] = df_projects['closingdate'].dt.year
df_projects['closingday'] = df_projects['closingdate'].dt.day
df_projects['closingweekday'] = df_projects['closingdate'].dt.weekday

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df_projects['totalamt'] = pd.to_numeric(df_projects['totalamt'].str.replace(',', ''))

ax = df_projects.groupby('approvalyear')['totalamt'].sum().plot(x='approvalyear', y='totalamt',
                                                                title='Total Amount Approved per Year')
ax.set_xlabel('year')
ax.set_ylabel('amount $')
plt.show()

