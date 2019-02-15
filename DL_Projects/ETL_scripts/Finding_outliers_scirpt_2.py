import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

reload(sys)
sys.setdefaultencoding('utf8')


# read in the projects data set and do basic wrangling
gdp = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/gdp_data.csv', skiprows=4)
gdp.drop(['Unnamed: 62', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)
population = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/population_data.csv',
                         skiprows=4)
population.drop(['Unnamed: 62', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)


# Reshape the data sets so that they are in long format
gdp_melt = gdp.melt(id_vars=['Country Name'],
                    var_name='year',
                    value_name='gdp')

# Use back fill and forward fill to fill in missing gdp values
gdp_melt['gdp'] = gdp_melt.sort_values('year').groupby('Country Name')['gdp'].fillna(method='ffill').\
    fillna(method='bfill')

population_melt = population.melt(id_vars=['Country Name'],
                                  var_name='year',
                                  value_name='population')

# Use back fill and forward fill to fill in missing population values
population_melt['population'] = population_melt.sort_values('year').groupby('Country Name')['population'].\
    fillna(method='ffill').fillna(method='bfill')

# merge the population and gdp data together into one data frame
df_country = gdp_melt.merge(population_melt, on=('Country Name', 'year'))

# filter data for the year 2016
df_2016 = df_country[df_country['year'] == '2016']

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
df_2016 = df_2016[~df_2016['Country Name'].isin(non_countries)]


# plot the data
x = list(df_2016['population'])
y = list(df_2016['gdp'])
text = df_2016['Country Name']

fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('GDP')
plt.ylabel('Population')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i],y[i]))
plt.show()


# TODO: Write a function that uses the Tukey rule to detect outliers in a dataframe column
# and then removes that entire row from the data frame. For example, if the United States
# is detected to be a GDP outlier, then remove the entire row of United States data.
# The function inputs should be a data frame and a column name.
# The output is a data_frame with the outliers eliminated

# HINT: Re-use code from the previous exercise


def tukey_rule(data_frame, column_name):
    data = data_frame[column_name]
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)

    IQR = Q3 - Q1

    max_value = Q3 + 1.5 * IQR
    min_value = Q1 - 1.5 * IQR

    return data_frame[(data_frame[column_name] < max_value) & (data_frame[column_name] > min_value)]


df_outlier_removed = df_2016.copy()

for column in ['population', 'gdp']:
    df_outlier_removed = tukey_rule(df_outlier_removed, column)

x = list(df_outlier_removed['population'])
y = list(df_outlier_removed['gdp'])
text = df_outlier_removed['Country Name']

fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(x, y)
plt.title("gdp vs population")
plt.xlabel("gdp")
plt.ylabel("Population")
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i], y[i]))
plt.show()
