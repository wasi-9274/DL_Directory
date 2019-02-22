import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

movies = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/movies.dat',
                     delimiter='::', header=None, names=['movie_id', 'movie', 'genre'],
                     dtype={'movie_id': object}, engine='python')
reviews = pd.read_csv('https://raw.githubusercontent.com/sidooms/MovieTweetings/master/latest/ratings.dat',
                      delimiter='::', header=None,
                      names=['user_id', 'movie_id', 'rating', 'timestamp'],
                      dtype={'movie_id': object, 'user_id': object, 'timestamp': object}, engine='python')
genres = []
for val in movies.genre:
    try:
        genres.extend(val.split('|'))
    except AttributeError:
        pass

genres = set(genres)
print("The number of genres is {}".format(len(genres)))

# pull date if it exists

create_date = lambda val: val[-5:-1] if val[-1] == ')' else np.nan

# apply the function to pull the date

movies['date'] = movies['movie'].apply(create_date)

# Return century of movie as a dummy column


def add_movie_year(val):
    if val[:2] == yr:
        return 1
    else:
        return 0


# Apply function

for yr in ['18', '19', '20']:
    movies[str(yr) + "00's"] = movies['date'].apply(add_movie_year)


# Function to split and return values for columns
def split_genres(val):
    try:
        if val.find(gene) >- 1:
            return 1
        else:
            return 0
    except AttributeError:
        return 0


# Apply function for each genre
for gene in genres:
    movies[gene] = movies['genre'].apply(split_genres)

print(movies.head())
print()

# adding date from the timestamp

change_timestamp = lambda val: datetime.datetime.fromtimestamp(int(val)).strftime('%Y-%m-%d %H:%M:%S')

reviews['date'] = reviews['timestamp'].apply(change_timestamp)

print(movies.head())
print(reviews.head())
print()

# now reviews and movies are the final dataframes with the necessary columns

reviews.to_csv('/home/wasi/Desktop/main_df/new_reviews.csv', index=False)
movies.to_csv('/home/wasi/Desktop/main_df/new_movies.csv', index=False)

