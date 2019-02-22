import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import progressbar
from IPython.display import HTML

movies = pd.read_csv("/home/wasi/ML_FOLDER/Udacity-DSND-master/Experimental Design & Recommandations/Recommendations/"
                     "1_Intro_to_Recommendations/movies_clean.csv")
reviews = pd.read_csv("/home/wasi/ML_FOLDER/Udacity-DSND-master/Experimental Design & Recommandations/Recommendations/"
                      "1_Intro_to_Recommendations/reviews_clean.csv")


# del movies['Unnamed: 0']
# del reviews['Unnamed: 0']

# print(movies.head())
# print(reviews.head())

user_items = reviews[['user_id', 'movie_id', 'rating']]
# print(user_items.head())

user_by_movie = user_items.groupby(['user_id', 'movie_id'])['rating'].max().unstack()


def movies_watched(user_id):
    '''
    INPUT:
    user_id - the user_id of an individual as int
    OUTPUT:
    movies - an array of movies the user has watched
    '''
    movies = user_by_movie.loc[user_id][user_by_movie.loc[user_id].isnull() == False].index.values

    return movies


def create_user_movie_dict():
    '''
    INPUT: None
    OUTPUT: movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids

    Creates the movies_seen dictionary
    '''
    n_users = user_by_movie.shape[0]
    movies_seen = dict()

    # Set up a progress bar
    cnter = 0
    bar = progressbar.ProgressBar(maxval=n_users+1, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    for user1 in range(1, n_users+1):

        # update progress bar
        cnter+=1
        bar.update(cnter)

        # assign list of movies to each user key
        movies_seen[user1] = movies_watched(user1)

    bar.finish()

    return movies_seen


movies_seen = create_user_movie_dict()


def create_movies_to_analyze(movies_seen, lower_bound=2):
    '''
    INPUT:
    movies_seen - a dictionary where each key is a user_id and the value is an array of movie_ids
    lower_bound - (an int) a user must have more movies seen than the lower bound to be added to the movies_to_analyze dictionary

    OUTPUT:
    movies_to_analyze - a dictionary where each key is a user_id and the value is an array of movie_ids

    The movies_seen and movies_to_analyze dictionaries should be the same except that the output dictionary has removed

    '''
    movies_to_analyze = dict()

    for user, movies in movies_seen.items():
        if len(movies) > lower_bound:
            movies_to_analyze[user] = movies
    return movies_to_analyze


movies_to_analyze = create_movies_to_analyze(movies_seen)






