import pandas as pd

df = pd.read_csv("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/gdp_data.csv", skiprows=4)
df.drop("Unnamed: 62", axis=1, inplace=True)
# print(df.head())

# print(df.isnull().sum())

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
# put the data set into long form instead of wide
df_melt = pd.melt(df,
                  id_vars=['Country Name',
                           'Country Code', 'Indicator Name', 'Indicator Code'],
                  var_name='year', value_name='GDP')

# print(df_melt.head())

df_melt['year'] = pd.to_datetime(df_melt['year'])
#
# print(df_melt['year'].dt.year.unique())
# print()


def plot_results(column_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    df_melt[(df_melt['Country Name'] == 'Afghanistan') |
            (df_melt['Country Name'] == 'Albania') |
            (df_melt['Country Name'] == 'Honduras')].groupby('Country Name').plot('year',
                                                                                  column_name,
                                                                                  legend=True,
                                                                                  ax=ax)
    ax.legend(labels=['Afghanistan', 'Albania', 'Honduras'])
    plt.show()


plot_results('GDP')

df_melt['GDP_filled'] = df_melt.groupby('Country Name')['GDP'].transform(lambda x: x.fillna(x.mean()))

print(df_melt.head())
plot_results('GDP_filled')

df_melt['GDP_ffill'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].\
    transform(lambda x: x.fillna(x.mean()))

print(df_melt.head())
plot_results('GDP_ffill')

# df_melt['GDP_bfill'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].fillna(method='bfill')
df_melt['GDP_bfill'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].\
    transform(lambda x: x.fillna(x.mean()))

plot_results('GDP_bfill')

df_melt['GDP_ff_bf'] = df_melt.sort_values('year').groupby('Country Name')['GDP'].fillna(method='ffill').\
    fillna(method='bfill')

# Check if any GDP values are null
df_melt['GDP_ff_bf'].isnull().sum()
