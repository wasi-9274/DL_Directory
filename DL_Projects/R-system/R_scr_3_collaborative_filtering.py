import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import progressbar

movies = pd.read_csv("/home/wasi/Desktop/main_df/new_movies.csv")
reviews = pd.read_csv("/home/wasi/Desktop/main_df/new_reviews.csv")
# print(reviews.shape)
# print(reviews.head())

user_items = reviews[['user_id', 'movie_id', 'rating']]
print(user_items.head())

user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
# print(user_by_movie)


def movies_watched(user_id):

    movies = user_by_movie.loc[user_id][user_by_movie.loc[user_id].isnull() == False].index.values

    return movies


# print(user_by_movie.shape[0])

n_users = user_by_movie.shape[0]
movies_seen = dict()
for user1 in range(1, n_users+1):
    movies_seen[user1] = movies_watched(user1)
# print()


def create_user_movie_dict():
    n_users = user_by_movie.shape[0]
    movies_seen = dict()

    # Set up a progress bar
    cnter = 0
    bar = progressbar.ProgressBar(maxval=n_users+1, widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                             progressbar.Percentage()])
    bar.start()

    for user1 in range(1, n_users+1):

        # update progress bar
        cnter += 1
        bar.update(cnter)

        # assign list of movies to each user key
        movies_seen[user1] = movies_watched(user1)

    bar.finish()

    return movies_seen


movies_seen = create_user_movie_dict()
# print(movies_seen)


def create_movies_to_analyze(movies_seen, lower_bound=2):
    movies_to_analyze = dict()

    for user, movies in movies_seen.items():
        if len(movies) > lower_bound:
            movies_to_analyze[user] = movies
    return movies_to_analyze


movies_to_analyze = create_movies_to_analyze(movies_seen)
# print(type(movies_to_analyze))


def compute_correlation(user1, user2):
    # Pull movies for each user
    movies1 = movies_to_analyze[user1]
    movies2 = movies_to_analyze[user2]

    # Find Similar Movies
    sim_movs = np.intersect1d(movies1, movies2, assume_unique=True)

    # Calculate correlation between the users
    df = user_by_movie.loc[(user1, user2), sim_movs]
    corr = df.transpose().corr().iloc[0, 1]
    return corr


results_compute_correlation = compute_correlation(2, 3)
print(results_compute_correlation)


def compute_euclidean_dist(user1, user2):
    # Pull movies for each user
    movies1 = movies_to_analyze[user1]
    movies2 = movies_to_analyze[user2]

    # Find Similar Movies
    sim_movs = np.intersect1d(movies1, movies2, assume_unique=True)

    # Calculate euclidean distance between the users
    df = user_by_movie.loc[(user1, user2), sim_movs]
    dist = np.linalg.norm(df.loc[user1] - df.loc[user2])

    return dist


results_compute_euclidean_dist = compute_euclidean_dist(2, 2)
print(results_compute_euclidean_dist)




