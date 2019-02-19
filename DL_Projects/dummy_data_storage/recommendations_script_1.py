import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

movies = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat',
                     delimiter='::', header=None, names=['movie_id', 'movie', 'genre'],
                     dtype={'movie_id': object}, engine='python')

reviews = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat',
                      delimiter='::', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      dtype={'movie_id': object, 'user_id': object, 'timestamp': object}, engine='python')

print(movies.head())
print(reviews.head())
print("The number of movies is {}".format(movies.shape[0]))
print("The number of ratings is {}".format(reviews.shape[0]))
print("The number of unique users is {}".format(reviews.user_id.nunique()))
print("The number of missing reviews is {}.".format(int(reviews.rating.isnull().mean()*reviews.shape[0])))
print("The average, minimum, and max ratings given are {}, {}, and {}, respectively."
      .format(np.round(reviews.rating.mean(), 0), reviews.rating.min(), reviews.rating.max()))

# number of different genres

genres = []
for val in movies.genre:
    try:
        genres.extend(val.split('|'))
    except AttributeError:
        pass

genres = set(genres)
print("The number of genre is {}".format(len(genres)))

# pulling the date if it exists

create_date = lambda val: val[-5: -1] if val[-1] == ')' else np.nan

movies['date'] = movies['movie'].apply(create_date)


def add_movie_year(val):
    if val[:2] == yr:
        return 1
    else:
        return 0


for yr in ['18', '19', '20']:
    movies[str(yr) + "00's"] = movies['date'].apply(add_movie_year)

print(movies.head())


def split_genres(val):
    try:
        if val.find(gene) > -1:
            return 1
        else:
            return 0
    except AttributeError:
        return 0


for gene in genres:
    movies[gene] = movies['genre'].apply(split_genres)

print(movies.head())

change_timestamp = lambda val: datetime.datetime.fromtimestamp(int(val)).strftime('%Y-%m-%d %H:%M:%S')
reviews['date'] = reviews['timestamp'].apply(change_timestamp)
print(reviews['date'])

reviews.to_csv('/home/wasi/Desktop/scrapy_projects/DL_Projects/dummy_data_storage/reviews_clean.csv', index=False)
movies.to_csv("/home/wasi/Desktop/scrapy_projects/DL_Projects/dummy_data_storage/movies_clean.csv", index=False)