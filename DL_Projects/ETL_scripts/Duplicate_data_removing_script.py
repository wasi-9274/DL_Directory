import pandas as pd
import datetime
import numpy as np

projects = pd.read_csv("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/data/projects_data.csv", dtype=str)
projects.drop("Unnamed: 56", axis=1, inplace=True)
projects['totalamt'] = pd.to_numeric(projects['totalamt'].str.replace(",", ""))
projects['countryname'] = projects['countryname'].str.split(';', expand=True)[0]
projects['boardapprovaldate'] = pd.to_datetime(projects['boardapprovaldate'])
#
# print()
#
# print(projects[projects['totalamt'] > 1000000000]['countryname'].nunique())

# print(projects[projects['countryname'].str.contains('Yugoslavia')])

republics = projects[(projects['boardapprovaldate'] < datetime.date(1992, 4, 27)) &
                     ((projects['countryname'].str.contains('Bosnia')) |
                      (projects['countryname'].str.contains('Croatia')) |
                      (projects['countryname'].str.contains('Kosovo')) |
                      (projects['countryname'].str.contains('Macedonia')) |
                      (projects['countryname'].str.contains('Montenegro')) |
                      (projects['countryname'].str.contains('Serbia')) |
                      (projects['countryname'].str.contains('Slovenia')))][['regionname',
                                                                            'countryname',
                                                                            'lendinginstr',
                                                                            'totalamt',
                                                                            'boardapprovaldate',
                                                                            'location',
                                                                            'GeoLocID',
                                                                            'GeoLocName',
                                                                            'Latitude',
                                                                            'Longitude',
                                                                            'Country',
                                                                            'project_name']].\
    sort_values('boardapprovaldate')

# print(republics.head())

yugoslavia = projects[(projects['countryname'].str.contains('Yugoslavia')) &
                      (projects['boardapprovaldate'] > datetime.date(1980, 2, 1)) &
                      (projects['boardapprovaldate'] < datetime.date(1989, 5, 23))][['regionname',
                                                                                     'countryname',
                                                                                     'lendinginstr',
                                                                                     'totalamt',
                                                                                     'boardapprovaldate',
                                                                                     'location',
                                                                                     'GeoLocID',
                                                                                     'GeoLocName',
                                                                                     'Latitude',
                                                                                     'Longitude',
                                                                                     'Country',
                                                                                     'project_name']].\
    sort_values('boardapprovaldate')

republic_unique_dates = republics['boardapprovaldate'].unique()

yugoslavia_unique_dates = yugoslavia['boardapprovaldate'].unique()

dates = np.append(republic_unique_dates, yugoslavia_unique_dates)

unique_dates, count = np.unique(dates, return_counts=True)
print(unique_dates, count)

for i in range(len(unique_dates)):
    if count[i] == 2:
        print(unique_dates[i])

print(pd.concat([yugoslavia[yugoslavia['boardapprovaldate'] == datetime.date(1983, 7, 26)],
           republics[republics['boardapprovaldate'] == datetime.date(1983, 7, 26)]]))
