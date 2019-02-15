import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycountry import countries
from collections import defaultdict
import sqlite3
sns.set()

# read in the projects data set and do basic wrangling
gdp = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/gdp_data.csv', skiprows=4)
gdp.drop(['Unnamed: 62', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)
population = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/population_data.csv',
                         skiprows=4)
population.drop(['Unnamed: 62', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)


# Reshape the data sets so that they are in long format
gdp_melt = gdp.melt(id_vars=['Country Name', 'Country Code'],
                    var_name='year',
                    value_name='gdp')

# Use back fill and forward fill to fill in missing gdp values
gdp_melt['gdp'] = gdp_melt.sort_values('year').groupby(['Country Name', 'Country Code'])['gdp'].fillna(method='ffill').fillna(method='bfill')

population_melt = population.melt(id_vars=['Country Name', 'Country Code'],
                                  var_name='year',
                                  value_name='population')

# Use back fill and forward fill to fill in missing population values
population_melt['population'] = population_melt.sort_values('year').groupby('Country Name')['population'].fillna(method='ffill').fillna(method='bfill')

# merge the population and gdp data together into one data frame
df_indicator = gdp_melt.merge(population_melt, on=('Country Name', 'Country Code', 'year'))

# filter out values that are not countries
non_countries = ['World',
                 'High income',
                 'OECD members',
                 'Post-demographic dividend',
                 'IDA & IBRD total',
                 'Low & middle income',
                 'Middle income',
                 'IBRD only',
                 'East Asia & Pacific',
                 'Europe & Central Asia',
                 'North America',
                 'Upper middle income',
                 'Late-demographic dividend',
                 'European Union',
                 'East Asia & Pacific (excluding high income)',
                 'East Asia & Pacific (IDA & IBRD countries)',
                 'Euro area',
                 'Early-demographic dividend',
                 'Lower middle income',
                 'Latin America & Caribbean',
                 'Latin America & the Caribbean (IDA & IBRD countries)',
                 'Latin America & Caribbean (excluding high income)',
                 'Europe & Central Asia (IDA & IBRD countries)',
                 'Middle East & North Africa',
                 'Europe & Central Asia (excluding high income)',
                 'South Asia (IDA & IBRD)',
                 'South Asia',
                 'Arab World',
                 'IDA total',
                 'Sub-Saharan Africa',
                 'Sub-Saharan Africa (IDA & IBRD countries)',
                 'Sub-Saharan Africa (excluding high income)',
                 'Middle East & North Africa (excluding high income)',
                 'Middle East & North Africa (IDA & IBRD countries)',
                 'Central Europe and the Baltics',
                 'Pre-demographic dividend',
                 'IDA only',
                 'Least developed countries: UN classification',
                 'IDA blend',
                 'Fragile and conflict affected situations',
                 'Heavily indebted poor countries (HIPC)',
                 'Low income',
                 'Small states',
                 'Other small states',
                 'Not classified',
                 'Caribbean small states',
                 'Pacific island small states']

# remove non countries from the data
df_indicator  = df_indicator[~df_indicator['Country Name'].isin(non_countries)]
df_indicator.reset_index(inplace=True, drop=True)

df_indicator.columns = ['countryname', 'countrycode', 'year', 'gdp', 'population']

# output the first few rows of the data frame
print(df_indicator.head())

################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################

# read in the projects data set with all columns type string

df_projects = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/projects_data.csv',
                          dtype=str)
df_projects.drop(['Unnamed: 56'], axis=1, inplace=True)

df_projects['countryname'] = df_projects['countryname'].str.split(';').str.get(0)

# set up the libraries and variables

country_not_found = [] # stores countries not found in the pycountry library
project_country_abbrev_dict = defaultdict(str) # set up an empty dictionary of string values

# TODO: iterate through the country names in df_projects.
# Create a dictionary mapping the country name to the alpha_3 ISO code
for country in df_projects['countryname'].drop_duplicates().sort_values():
    try:
        # TODO: look up the country name in the pycountry library
        # store the country name as the dictionary key and the ISO-3 code as the value
        project_country_abbrev_dict[country] = countries.lookup(country).alpha_3
    except:
        # If the country name is not in the pycountry library, then print out the country name
        # And store the results in the country_not_found list
        country_not_found.append(country)

# run this code cell to load the dictionary

country_not_found_mapping = {'Co-operative Republic of Guyana': 'GUY',
                             'Commonwealth of Australia':'AUS',
                             'Democratic Republic of Sao Tome and Prin':'STP',
                             'Democratic Republic of the Congo':'COD',
                             'Democratic Socialist Republic of Sri Lan':'LKA',
                             'East Asia and Pacific':'EAS',
                             'Europe and Central Asia': 'ECS',
                             'Islamic  Republic of Afghanistan':'AFG',
                             'Latin America':'LCN',
                             'Caribbean':'LCN',
                             'Macedonia':'MKD',
                             'Middle East and North Africa':'MEA',
                             'Oriental Republic of Uruguay':'URY',
                             'Republic of Congo':'COG',
                             "Republic of Cote d'Ivoire":'CIV',
                             'Republic of Korea':'KOR',
                             'Republic of Niger':'NER',
                             'Republic of Kosovo':'XKX',
                             'Republic of Rwanda':'RWA',
                             'Republic of The Gambia':'GMB',
                             'Republic of Togo':'TGO',
                             'Republic of the Union of Myanmar':'MMR',
                             'Republica Bolivariana de Venezuela':'VEN',
                             'Sint Maarten':'SXM',
                             "Socialist People's Libyan Arab Jamahiriy":'LBY',
                             'Socialist Republic of Vietnam':'VNM',
                             'Somali Democratic Republic':'SOM',
                             'South Asia':'SAS',
                             'St. Kitts and Nevis':'KNA',
                             'St. Lucia':'LCA',
                             'St. Vincent and the Grenadines':'VCT',
                             'State of Eritrea':'ERI',
                             'The Independent State of Papua New Guine':'PNG',
                             'West Bank and Gaza':'PSE',
                             'World':'WLD'}

project_country_abbrev_dict.update(country_not_found_mapping)

df_projects['countrycode'] = df_projects['countryname'].apply(lambda x: project_country_abbrev_dict[x])

df_projects['boardapprovaldate'] = pd.to_datetime(df_projects['boardapprovaldate'])

df_projects['year'] = df_projects['boardapprovaldate'].dt.year.astype(str).str.slice(stop=4)

df_projects['totalamt'] = pd.to_numeric(df_projects['totalamt'].str.replace(',', ""))

df_projects = df_projects[['id', 'countryname', 'countrycode', 'totalamt', 'year']]

print(df_projects.head())

################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################

# TODO: merge the projects and indicator data frames together using countrycode and year as common keys
# Use a left join so that all projects are returned even if the country/year combination does not have
# indicator data

df_merged = df_projects.merge(df_indicator, how='left', on=['countrycode', 'year'])

print(df_merged.head(30))

################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################
# Run this code to check your work

print(df_merged[(df_merged['year'] == '2017') & (df_merged['countryname_y'] == 'Jordan')])

################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################

# TODO: Output the df_merged data frame as a json file
# HINT: Pandas has a to_json() method
# HINT: use orient='records' to get one of the more common json formats
# HINT: be sure to specify the name of the json file you want to create as the first input into to_json
df_merged.to_json('/home/wasi/Desktop/junk/countrydata.json', orient='records')

################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################

# TODO: Output the df_merged data frame as a csv file
# HINT: The to_csv() method is similar to the to_json() method.
# HINT: If you do not want the data frame indices in your result, use index=False
df_merged.to_csv('/home/wasi/Desktop/junk/countrydata.csv', index=False)

################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################


#
# # connect to the database
# # the database file will be worldbank.db
# # note that sqlite3 will create this database file if it does not exist already
# conn = sqlite3.connect('worldbank.db')
#
# # TODO: output the df_merged dataframe to a SQL table called 'merged'.
# # HINT: Use the to_sql() method
# # HINT: Use the conn variable for the connection parameter
# # HINT: You can use the if_exists parameter like if_exists='replace' to replace a table if it already exists
#
# df_merged.to_sql('merged', con=conn, if_exists='replace', index=False)


################################### END OF THE PART OF CODE NEXT STARTS NEW CODE ######################################

