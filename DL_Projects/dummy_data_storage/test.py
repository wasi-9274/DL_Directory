import pandas as pd
import numpy as np

df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo", "bar", "bar", "bar", "bar"],
                   "B": ["one", "one", "one", "two", "two", "one", "one", "two", "two"],
                   "C": ["small", "large", "large", "small", "small", "large", "small", "small", "large"],
                   "D": [1, 2, 2, 3, 3, 4, 5, 6, 7], "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

print(df)

table = pd.pivot_table(df, values='D', index=['A', 'B'],
                       columns=['C'], aggfunc=np.sum)
print("=" * 30)

table2 = pd.pivot_table(df, values='D', index=['A', 'B'],
                        columns=['C'], aggfunc=np.sum,
                        fill_value=0)

print(table.head())
print("=" * 30)
print(table2.head())

result = pd.read_csv("/home/wasi/ML_FOLDER/Udacity-DSND-master/Experimental Design & Recommandations/"
                     "Recommendations/1_Intro_to_Recommendations/reviews_clean.csv")
print(result.head(30))


print()


