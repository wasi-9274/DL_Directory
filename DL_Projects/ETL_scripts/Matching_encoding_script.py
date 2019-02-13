import pandas as pd

df = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/9_encodings_exercise/mystery.csv',
                 skiprows=22, encoding='utf-8')
print(df.head())
print()

from encodings.aliases import aliases

aliases_values = set(aliases.values())

for encoding in set(aliases.values()):
    try:
        df = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/9_encodings_exercise/mystery.csv',
                         skiprows=22)
        print('successful', encoding)
    except Exception as e:
        print(e)


import chardet

with open("/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/9_encodings_exercise/mystery.csv", 'rb') as \
        file_reader:
    print(chardet.detect(file_reader.read()))

df = pd.read_csv('/home/wasi/ML_FOLDER/DSND_Term2-master/lessons/ETLPipelines/9_encodings_exercise/mystery.csv',
                 skiprows=22, encoding='UTF-16')
print(df.head())
