import pandas as pd
import numpy as np
import re

projects = pd.read_csv("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/projects_data.csv", dtype=str)
projects.drop("Unnamed: 56", axis=1, inplace=True)
projects['totalamt'] = pd.to_numeric(projects['totalamt'].str.replace(",", ""))
# print(projects['totalamt'])
projects['countryname'] = projects['countryname'].str.split(",", expand=True)[0]
# print(projects['countryname'])
projects['boardapprovaldate'] = pd.to_datetime(projects['boardapprovaldate'])

sector = projects.copy()
sector = sector[['project_name', 'lendinginstr', 'sector1', 'sector2', 'sector3', 'sector4', 'sector5', 'sector',
                 'mjsector1', 'mjsector2', 'mjsector3', 'mjsector4', 'mjsector5',
                 'mjsector', 'theme1', 'theme2', 'theme3', 'theme4', 'theme5', 'theme ',
                 'goal', 'financier', 'mjtheme1name', 'mjtheme2name', 'mjtheme3name',
                 'mjtheme4name', 'mjtheme5name']]

# print(sector.head())

# shows the percentage of each variable that is null.

# print(100 * sector.isnull().sum() / sector.shape[0])
# print()
# uniquesector1 = list(sector['sector1'].sort_values().unique())
# print(uniquesector1)
# print(len(uniquesector1))

# print(sector['sector1'])

sector['sector1'] = sector['sector1'].replace('!$!0', np.nan)

sector['sector1'] = sector['sector1'].replace('!.+', '', regex=True)

sector['sector1'] = sector['sector1'].replace('^(\(Historic\))', '', regex=True)

# print('Number of unique sectors after cleaning:', len(list(sector['sector1'].unique())))
# print('Percentage of null values after cleaning:', 100 * sector['sector1'].
#       isnull().sum() / sector['sector1'].shape[0])

# print(sector['sector'])

sector.loc[:, 'sector1_aggregates'] = sector['sector1']

# print()

sector.loc[sector['sector1_aggregates'].str.contains('Energy', re.IGNORECASE).replace(np.nan, False),
           'sector1_aggregates'] = 'Energy'
sector.loc[sector['sector1_aggregates'].str.contains('Transportation', re.IGNORECASE).replace(np.nan,
                                                                                              False),
           'sector1_aggregates'] = 'Transportation'

# print('Number of unique sectors after cleaning:', len(list(sector['sector1_aggregates'].unique())))

# TODO: Create dummy variables from the sector1_aggregates data. Put the results into a dataframe called dummies
# Hint: Use the get_dummies method
dummies = pd.DataFrame(pd.get_dummies(sector['sector1_aggregates']))
print(dummies)

# TODO: Filter the projects data for the totalamt, the year from boardapprovaldate, and the dummy variables
projects['year'] = projects['boardapprovaldate'].dt.year
df = projects[['totalamt', 'year']]
df_final = pd.concat([df, dummies], axis=1)

print(df_final.head())