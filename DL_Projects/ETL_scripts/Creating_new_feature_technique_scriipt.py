import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

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
df_2016.reset_index(inplace=True, drop=True)
print(df_2016.head(30))

df_2016['gdppercapita'] = df_2016['gdp'] / df_2016['population']
print(df_2016.head(20))

# TODO: Fill out the create_multiples function.
# The create_multiples function has two inputs. A floating point number and an integer.
# The output is a list of multiples of the input b starting from the square of b and ending at b^k.


def create_multiples(b, k):

    new_features = []

    # TODO: use a for loop to make a list of multiples of b: ie b^2, b^3, b^4, etc... until b^k
    for i in range(2, k+1):
        new_features.append(b ** i)

    return new_features

# TODO: Fill out the column_name_generator function.
# The function has two inputs: a string representing a column name and an integer k.
# The 'k' variable is the same as the create_multiples function.
# The output should be a list of column names.
# For example if the inputs are ('gdp', 4) then the output is a list of strings ['gdp2', 'gdp3', gdp4']


def column_name_generator(colname, k):

    col_names = []
    for i in range(2, k+1):
        col_names.append('{}{}'.format(colname, i))
    return col_names

# TODO: Fill out the concatenate_features function.
# The function has three inputs. A dataframe, a column name represented by a string, and an integer representing
# the maximum power to create when engineering features.

# If the input is (df_2016, 'gdp', 3), then the output will be the df_2016 dataframe with two new columns
# One new column will be 'gdp2' ie gdp^2, and then other column will be 'gdp3' ie gdp^3.

# HINT: There may be more than one way to do this.
# The TODOs in this section point you towards one way that works


def concatenate_features(df, column, num_columns):

    # TODO: Use the pandas apply() method to create the new features. Inside the apply method, you
    # can use a lambda function with the create_mtuliples function
    new_features = df[column].apply(lambda x: create_multiples(x, num_columns))

    # TODO: Create a dataframe from the new_features variable
    # Use the column_name_generator() function to create the column names

    # HINT: In the pd.DataFrame() method, you can specify column names inputting a list in the columns option
    # HINT: Using new_features.tolist() might be helpful
    new_features_df = pd.DataFrame(new_features.tolist(), columns=column_name_generator(column, num_columns))

    # TODO: concatenate the original date frame in df with the new_features_df dataframe
    # return this concatenated dataframe
    return pd.concat([df, new_features_df], axis=1)


print(concatenate_features(df_2016, 'gdp', 4))
