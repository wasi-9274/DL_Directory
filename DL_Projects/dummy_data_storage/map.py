import numpy as np
import pandas as pd
df = pd.read_csv("/home/wasi/Desktop/test.csv")

cus_func = lambda x: {'female': 0, 'male': 1}.get(x)

df['re_used'] = df['gender'].apply(cus_func)
print(df)
