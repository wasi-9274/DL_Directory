import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.linear_model import LinearRegression

reload(sys)
sys.setdefaultencoding('utf8')


gdp = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/gdp_data.csv', skiprows=4)
gdp.drop(['Unnamed: 62', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)
population = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/population_data.csv',
                         skiprows=4)
population.drop(['Unnamed: 62', 'Country Code', 'Indicator Name', 'Indicator Code'], inplace=True, axis=1)

gdp_melt = gdp.melt(id_vars=['Country Name'],
                    var_name='year',
                    value_name='gdp')

# print(gdp_melt)

gdp_melt['gdp'] = gdp_melt.sort_values('year').groupby('Country Name')['gdp'].\
    fillna(method='ffill').fillna(method='bfill')

# print(gdp_melt)

population_melt = population.melt(id_vars=['Country Name'],
                                  var_name='year',
                                  value_name='population')


population_melt['population'] = population_melt.sort_values('year').groupby('Country Name')['population'].\
    fillna(method='ffill').fillna(method='bfill')

df_country = gdp_melt.merge(population_melt, on=('Country Name', 'year'))

df_2016 = df_country[df_country['year'] == '2016']

# print(df_2016)

# print(population_melt)
# print(gdp_melt)

df_2016.plot('population', kind='box')
df_2016.plot('gdp', kind='box')
# plt.show()

# Use the Tukey rule to determine what values of the population data are outliers for the year 2016.
# The Tukey rule finds outliers in one-dimension. The steps are:
#
#     Find the first quartile (ie .25 quantile)
#     Find the third quartile (ie .75 quantile)
#     Calculate the inter-quartile range (Q3 - Q1)
#     Any value that is greater than Q3 + 1.5 * IQR is an outlier
#     Any value that is less than Qe - 1.5 * IQR is an outlier

# TODO: Filter the data for the year 2016 and put the results in the population_2016 variable. You only need
# to keep the Country Name and population columns
population_2016 = df_2016[['Country Name', 'population']]


# TODO: Calculate the first quartile of the population values
# HINT: you can use the pandas quantile method
Q1 = population_2016['population'].quantile(0.25)

# TODO: Calculate the third quartile of the population values
Q3 = population_2016['population'].quantile(0.75)

# TODP: Calculate the interquartile range Q3 - Q1
IQR = Q3 - Q1

# print(IQR)

# TODO: Calculate the maximum value and minimum values according to the Tukey rule
# max_value is Q3 + 1.5 * IQR while min_value is Q1 - 1.5 * IQR
max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR

# print(max_value)
# print(min_value)

# TODO: filter the population_2016 data for population values that are greater than max_value or less than min_value
population_outliers = population_2016[(population_2016['population'] > max_value) | (population_2016['population'] <
                                                                                     min_value)]
# print(population_outliers)

# TODO: Filter the data for the year 2016 and put the results in the population_2016 variable. You only need
# to keep the Country Name and population columns
gdp_2016 = df_2016[['Country Name', 'gdp']]

# TODO: Calculate the first quartile of the population values
# HINT: you can use the pandas quantile method
Q1 = gdp_2016['gdp'].quantile(0.25)

# TODO: Calculate the third quartile of the population values
Q3 = gdp_2016['gdp'].quantile(0.75)

# TODP: Calculate the interquartile range Q3 - Q1
IQR = Q3 - Q1

# TODO: Calculate the maximum value and minimum values according to the Tukey rule
# max_value is Q3 + 1.5 * IQR while min_value is Q1 - 1.5 * IQR
max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR

# TODO: filter the population_2016 data for population values that are greater than max_value or less than min_value
gdp_outliers = gdp_2016[(gdp_2016['gdp'] > max_value) | (gdp_2016['gdp'] < min_value)]
# print(gdp_outliers)

# TODO: remove the rows from the data that have Country Name values in the non_countries list
# Store the filter results back into the df_2016 variable

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
# print(df_2016)

# TODO: Re-rerun the Tukey code with this filtered data to find population outliers

# TODO: Filter the data for the year 2016 and put the results in the population_2016 variable. You only need
# to keep the Country Name and population columns
population_2016 = df_2016[['Country Name', 'population']]

# TODO: Calculate the first quartile of the population values
# HINT: you can use the pandas quantile method
Q1 = population_2016['population'].quantile(0.25)

# TODO: Calculate the third quartile of the population values
Q3 = population_2016['population'].quantile(0.75)

# TODO: Calculate the interquartile range Q3 - Q1
IQR = Q3 - Q1

# TODO: Calculate the maximum value and minimum values according to the Tukey rule
# max_value is Q3 + 1.5 * IQR while min_value is Q1 - 1.5 * IQR
max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR

# TODO: filter the population_2016 data for population values that are greater than max_value or less than min_value
population_outliers = population_2016[(population_2016['population'] > max_value) | (population_2016['population']
                                                                                     < min_value)]

# print(population_2016)

# TODO: Filter the data for the year 2016 and put the results in the population_2016 variable. You only need
# to keep the Country Name and population columns
gdp_2016 = df_2016[['Country Name','gdp']]

# TODO: Calculate the first quartile of the population values
# HINT: you can use the pandas quantile method
Q1 = gdp_2016['gdp'].quantile(0.25)

# TODO: Calculate the third quartile of the population values
Q3 = gdp_2016['gdp'].quantile(0.75)

# TODO: Calculate the interquartile range Q3 - Q1
IQR = Q3 - Q1

# TODO: Calculate the maximum value and minimum values according to the Tukey rule
# max_value is Q3 + 1.5 * IQR while min_value is Q1 - 1.5 * IQR
max_value = Q3 + 1.5 * IQR
min_value = Q1 - 1.5 * IQR

# TODO: filter the population_2016 data for population values that are greater than max_value or less than min_value
gdp_outliers = gdp_2016[(gdp_2016['gdp'] > max_value) | (gdp_2016['gdp'] < min_value)]
# print(gdp_outliers)

# TODO: Find country names that are in both the population_outliers and the gdp_outliers
# print(list(set(population_outliers['Country Name']).intersection(gdp_outliers['Country Name'])))

# TODO: Find country names that are in the population outliers list but not the gdp outliers list
# HINT: Python's set() and list() methods should be helpful

print(list(set(population_outliers['Country Name']) - set(gdp_outliers['Country Name'])))

# TODO: Find country names that are in the gdp outliers list but not the population outliers list
# HINT: Python's set() and list() methods should be helpful

print(list(set(gdp_outliers['Country Name']) - set(population_outliers['Country Name'])))

# run the code cell below

x = list(df_2016['population'])
y = list(df_2016['gdp'])
text = df_2016['Country Name']

fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('population')
plt.ylabel('GDP')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i], y[i]))
plt.show()

# Run the code below to see the results
df_no_large = (df_2016['Country Name'] != 'United States') & (df_2016['Country Name'] != 'India')\
              & (df_2016['Country Name'] != 'China')
x = list(df_2016[df_no_large]['population'])
y = list(df_2016[df_no_large]['gdp'])
text = df_2016[df_no_large]['Country Name']

fig, ax = plt.subplots(figsize=(15, 10))
ax.scatter(x, y)
plt.title('GDP vs Population')
plt.xlabel('population')
plt.ylabel('GDP')
for i, txt in enumerate(text):
    ax.annotate(txt, (x[i], y[i]))
plt.show()

# fit a linear regression model on the population and gdp data
model = LinearRegression()
model.fit(df_2016['population'].values.reshape(-1, 1), df_2016['gdp'].values.reshape(-1, 1))

# plot the data along with predictions from the linear regression model
inputs = np.linspace(1, 2000000000, num=50)
predictions = model.predict(inputs.reshape(-1, 1))

df_2016.plot('population', 'gdp', kind='scatter')
plt.plot(inputs, predictions)
print(model.predict(1000000000))
