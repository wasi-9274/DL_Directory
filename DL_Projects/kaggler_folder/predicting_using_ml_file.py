from collections import Counter
import gc
gc.enable()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow.python.data import Dataset
tf.logging.set_verbosity(tf.logging.ERROR)


# Okay, let's load in our data_sets
raw_train_df = pd.read_csv("/home/wasi/Desktop/scrapy_projects/DL_Projects/Titanic_folder/train.csv")
raw_test_df = pd.read_csv("/home/wasi/Desktop/scrapy_projects/DL_Projects/Titanic_folder/test.csv")
example_submission_df = pd.read_csv("/home/wasi/Desktop/scrapy_projects/DL_Projects/Titanic_folder/"
                                    "gender_submission.csv")

train_df = raw_train_df.copy(deep=True)
test_df = raw_test_df.copy(deep=True)
train_test_lst = [train_df, test_df]
train_df[train_df['Fare'] == 0]

# Both train and test datasets appear to have NaN values this could cause problems for our model, so let's
# look at what is missing and how much

print(train_df.isnull().sum())
print("Total indiviuals in train set is : {}".format(len(train_df)))
#
# print(test_df.isnull().sum())
# print("Total individuals in test set is: {}".format(len(test_df)))

# The huge amount of missing Cabin data is worrying, but let's see if it has any predictive power
#  before figuring out what to do

cabin_df = train_df[train_df['Cabin'].notnull()]
cabin_df = cabin_df.assign(deck_level=pd.Series([entry[:1] for entry in cabin_df['Cabin']]).values)
print(cabin_df.head())
print("Survival chances based on deck level:")
print(round(cabin_df.groupby(['deck_level'])['Survived'].mean() * 100))


def process_deck_lst(train_test_lst):
    new = []
    for dataset in train_test_lst:
        dataset = dataset.copy(True)
        dataset = dataset.assign(deck_level=pd.Series([[entry[:1]] if not pd.isnull(entry)
                                                       else
                                                       'U'
                                                       for entry in dataset['Cabin']]))
        dataset = dataset.drop(['Cabin'], axis=1)
        new.append(dataset)
    return new


train_df, test_df = process_deck_lst(train_test_lst)

print(set(train_df['Embarked']))
print("Survival chances based on embarcation: ")
print(train_df.groupby(['Embarked'])['Survived'].mean())

train_df[['Embarked']] = train_df[['Embarked']].fillna("N")
print(set(train_df['Embarked']))
print(train_df.isnull().sum())

print(test_df[test_df['Fare'].isnull()])


Pclass_Fare_grouping = test_df.groupby(["Pclass"])['Fare']
train_df.groupby(['Pclass', pd.cut(train_df['Fare'],
                                   np.arange(0, 701, 5))]).size().unstack(0).plot.bar(stacked=True,
                                                                                      title='Fare histogram '
                                                                                            'grouped by Pclass')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
print("Mean Fare for each Pclass:")
print(Pclass_Fare_grouping.mean())
print("Median Fare for each Pclass:")
print(Pclass_Fare_grouping.median())


test_df[['Fare']] = test_df[['Fare']].fillna(Pclass_Fare_grouping.median()[3])
print(test_df[test_df["PassengerId"] == 1044])
print(test_df.isnull().sum())

ax = train_df[["Age"]].plot(kind="hist", bins=20)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

train_df.groupby(['Survived', pd.cut(train_df['Age'], np.arange(0, 100, 5))]).size().unstack(0).\
    plot.bar(stacked=False, alpha=0.85)
_ = plt.title("Age histogram grouped by survival")
plt.show()

train_df.groupby(['Survived', 'Sex', pd.cut(train_df['Age'], np.arange(0, 100, 5))]).size().\
    unstack(0).plot.bar(stacked=True, alpha=0.75)
plt.title("Age histogram grouped by survival and gender")
plt.tight_layout()
plt.show()

train_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in train_df['Name']]
print("Train set titles (and counts):")
print(Counter(train_titles))

print("\nTest set titles (and counts):")
test_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in test_df['Name']]
print(Counter(test_titles))

print("\n===============================")

age_missing_train_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in train_df[train_df['Age'].
    isnull()]['Name']]
print("\nTrain set titles (and counts) with missing ages:")
print(Counter(age_missing_train_titles))

age_missing_test_titles = [name.split(',')[1].lstrip(' ').split('.')[0] for name in test_df[test_df['Age'].
    isnull()]['Name']]
print("\nTest set titles (and counts) with missing ages:")
print(Counter(age_missing_test_titles))




