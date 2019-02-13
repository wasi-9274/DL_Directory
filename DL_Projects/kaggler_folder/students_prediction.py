import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
sns.set()


print(os.listdir("/home/wasi/Desktop/kaggle/students_predict/"))

df = pd.read_csv('/home/wasi/Desktop/kaggle/students_predict/student-mat.csv')

# checking the columns
# print(df.head())

# checking the tail of df with how many rows are available
# print(df.tail())

# checking the mean, max, std deviation and min max few important statistics
# print(df.describe())

# checking the shape
# print(df.shape)

# checking the datatypes and other info of the dataset
# print(df.info())

# checking if any columns has null values or empty values
# print(df.isnull().sum())

# looking at the first period grades of the students
plt.hist(x=df['G1'])
plt.show()

# looking at the second period grades of the students with an outlier
plt.hist(x=df['G2'])
plt.show()

# looking at the third period grades of the students
plt.hist(x=df['G3'])
plt.show()

# looking at all the columns of the data frame
# df.hist()
# plt.show()


# looking the co relation between the columns
# corr = df.corr()
# sns.heatmap(corr, xticklabels=corr.columns.values,
#             yticklabels=corr.columns.values)
# plt.show()

# print(df.cov())

# ploting the scater plot for grades
# df.plot(x="G2", y="G3", style='o')
# plt.show()

# calculating the exact age and getting the year of the birth
year = pd.datetime.now()
date_t = year.year-df['age']
df['date_t'] = date_t
# print(df)


# normalizing the data

x = df[['G1']].values.astype(float)
# print(x)

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df['G1_new'] = pd.DataFrame(x_scaled)
# print(df)

X = df.iloc[:, 30:33]
Y = df.iloc[:, 30]
# print(X)
# print(Y)

# data taken of validation of the data
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
model = GaussianNB()

scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

msg = "%f (%f)" % (cv_results.mean(), cv_results.std())
print(msg)

# data to train and test the model

# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
#
# model.fit(X_train, Y_train)
#
# Y_pred = model.predict(X_test)

# Summary of the predictions made by the classifier

# print(classification_report(Y_test, Y_pred))
#
# print(confusion_matrix(Y_test, Y_pred))

# Accuracy score
# print("ACC: ", accuracy_score(Y_pred, Y_test))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


def larger_model():
    model = Sequential()
    model.add(Dense(257, input_dim=3, kernel_initializer="normal",
                    activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(128, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, kernel_initializer='normal', activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_dim =3, kernel_initializer="normal",
                    activation="relu"))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam',
                  metrics=['accuracy'])
    return model


def wider_model():
    # create model
    model = Sequential()
    model.add(Dense(20, input_dim=3, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


np.random.seed(seed)
estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasRegressor(build_fn=wider_model, epochs=1000, batch_size=5, verbose=1)))
pipeline = Pipeline(estimators)
kfold = KFold(n_splits=10, random_state=seed)
results = cross_val_score(pipeline, X_train, y_train, cv=kfold, n_jobs=1)
print("Larger: %.2f (%.2f) MSE" % (results.mean(), results.std()))


# model = LinearRegression()
# model_1 = LogisticRegression()
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
df3 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df3)







