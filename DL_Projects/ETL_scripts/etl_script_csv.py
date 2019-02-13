import pandas as pd
import numpy as np
import json


# EXTRACTING THE DATA CSV

# df = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/1_csv_exercise/population_data.csv',
#                  skiprows=4)
#
# df_1 = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/1_csv_exercise/projects_data.csv',
#                    dtype=str)
#
# print(df.head())
# print(df_1.head())
#
# print(df.isnull().sum())
# print(df_1.isnull().sum())

# print(df.isnull().sum(axis=1))
#
# df = df.drop('Unnamed: 62', axis=1)
# print(df.head())

# print(df[df.isnull().any(axis=1)])

# EXTRACTING THE DATA FROM JSON AND XML

# def print_lines(n, file_name):
#     f = open(file_name)
#     for i in range(10):
#         print(f.readline())
#     f.close()
#

# print_lines(1, '/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/2_extract_exercise/population_data.json')

##################### FINISH OF THE METHOD #######################

# df_json = pd.read_json('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/2_extract_exercise/'
#                        'population_data.json', orient='records')

# print(df_json.head())

##################### FINISH OF THE METHOD #######################

# Other ways to read in json

# with open("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/2_extract_exercise/population_data.json") as f:
#     json_data = json.load(f)

# print(json_data[0])
# print('\n')
#
# print(json_data[0]['Country Name'])
# print(json_data[0]['Country Code'])

##################### FINISH OF THE METHOD #######################

# NEXT WORKING WITH THE XML FILES

# print_lines(15, '/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/2_extract_exercise/population_data.xml')

# data extracting from xml using beautifulsoup

# from bs4 import BeautifulSoup
#
# with open("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/2_extract_exercise/population_data.xml") as fp:
#     # lxml is the parser type
#     soup = BeautifulSoup(fp, "lxml")
#
# i = 0
# for record in soup.find_all('record'):
#     i += 1
#     for record in record.find_all('field'):
#         print(record['name'], ': ', record.txt)
#     print()
#     if i == 5:
#         break
# print("Successfully Done")

##################### FINISH OF THE METHOD #######################

# EXTRACTING THE DATA USING API PROVIDED

# Example Indicators API
#
# Run the code example below to request data from the World Bank Indicators API. According to
# the documntation, you format your request url like so:
#
# http://api.worldbank.org/v2/countries/ + list of country abbreviations separated by ;
# + /indicators/ + indicator name + ? + options
#
# where options can include
#
# per_page - number of records to return per page
# page - which page to return - eg if there are 5000 records and 100 records per page
# date - filter by dates
#
# format - json or xml

import requests

# url = 'http://api.worldbank.org/v2/countries/br;cn;us;de/indicators/SP.POP.TOTL/?format=json&per_page=1000'
# r = requests.get(url)
# print(r.json())
#
# print(pd.DataFrame(r.json()[1]))

# To find the indicator code, first search for the indicator here: https://data.worldbank.org Click on the indicator
# name. The indicator code is in the url. For example,
# the indicator code for total population is SP.POP.TOTL. The link is https://data.worldbank.org/indicator/SP.RUR.TOTL.

url = 'http://api.worldbank.org/v2/countries/ch/indicators/SP.RUR.TOTL/?format=json&date=1995:2001'

r = requests.get(url)

# print(r.json())


##################### FINISH OF THE METHOD #######################


# COMBINING THE DATA

df_rural = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/5_combinedata_ex'
                       'ercise/rural_population_percent.csv', skiprows=4)
df_electricity = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/5_combinedata_'
                             'exercise/electricity_access_percent.csv', skiprows=4)

df_rural.drop('Unnamed: 62', axis=1, inplace=True)
df_electricity.drop("Unnamed: 62", axis=1, inplace=True)

df = pd.concat([df_rural, df_electricity])

# df.to_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/5_combinedata_exercise/df_rural.csv',
#           index=False)

# TODO: merge the data sets together according to the instructions. First, use the
# melt method to change the formatting of each data frame so that it looks like this:
# Country Name, Country Code, Year, Rural Value
# Country Name, Country Code, Year, Electricity Value

# print(df.head())
df_rural_melt = pd.melt(df_rural,
                        id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                        var_name='Year', value_name='Electricity_Value')
df_electricity_melt = pd.melt(df_electricity,
                              id_vars=['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'],
                              var_name='Year', value_name='Rural_Value')

df_rural_melt.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)
df_electricity_melt.drop(['Indicator Name', 'Indicator Code'], axis=1, inplace=True)

df_merge = df_rural_melt.merge(df_electricity_melt, how='outer',
                               on=['Country Name', 'Country Code', 'Year'])
# print(df_merge.head(100))

# print(df_rural_melt.head())
# print(df_electricity_melt.head())

##################### FINISH OF THE METHOD #######################















