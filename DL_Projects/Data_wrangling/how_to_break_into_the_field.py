import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/CRISP_DM/survey_results_public.csv")
# print(df.head())

df_2 = pd.read_csv("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/CRISP_DM/survey_results_schema.csv")
print(df_2.head())
print(list(df_2[df_2.Column == "CousinEducation"]["Question"]))


study = df['CousinEducation'].value_counts().reset_index()
print(study.head())

print("Nothing")
