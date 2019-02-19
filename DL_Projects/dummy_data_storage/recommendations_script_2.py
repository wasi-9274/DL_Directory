import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# read in the datasets

movies = pd.read_csv("/home/wasi/ML_FOLDER/Udacity-DSND-master/Experimental Design & Recommandations/Recommendations/"
                     "1_Intro_to_Recommendations/movies_clean.csv")
reviews = pd.read_csv("/home/wasi/ML_FOLDER/Udacity-DSND-master/Experimental Design & Recommandations/Recommendations/"
                      "1_Intro_to_Recommendations/reviews_clean.csv")
# del movies['Unnamed: 0']
# del reviews['Unnamed: 0']


def create_ranked_df(movies, reviews):
    '''
    INPUT
    movies - the movies dataframe
    reviews - the reviews dataframe

    OUTPUT
    ranked_movies - a dataframe with movies that are sorted by highest avg rating, more reviews,
                    then time, and must have more than 4 ratings
    '''

    # Pull the average ratings and number of ratings for each movie
    movie_ratings = reviews.groupby('movie_id')['rating']
    avg_ratings = movie_ratings.mean()
    num_ratings = movie_ratings.count()
    last_rating = pd.DataFrame(reviews.groupby('movie_id').max()['date'])
    last_rating.columns = ['last_rating']

    # Add Dates
    rating_count_df = pd.DataFrame({'avg_rating': avg_ratings, 'num_ratings': num_ratings})
    rating_count_df = rating_count_df.join(last_rating)

    # merge with the movies dataset
    movie_recs = movies.set_index('movie_id').join(rating_count_df)

    # sort by top avg rating and number of ratings
    ranked_movies = movie_recs.sort_values(['avg_rating', 'num_ratings', 'last_rating'], ascending=False)

    # for edge cases - subset the movie list to those with only 5 or more reviews
    ranked_movies = ranked_movies[ranked_movies['num_ratings'] > 4]

    return ranked_movies


def popular_recommendations(user_id, n_top, ranked_movies):
    '''
    INPUT:
    user_id - the user_id (str) of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time

    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''

    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies


# Top 20 movies recommended for id 1
ranked_movies = create_ranked_df(movies, reviews) # only run this once - it is not fast

recs_20_for_1 = popular_recommendations('1', 20, ranked_movies)

# Top 5 movies recommended for id 53968
recs_5_for_53968 = popular_recommendations('53968', 5, ranked_movies)

# Top 100 movies recommended for id 70000
recs_100_for_70000 = popular_recommendations('70000', 100, ranked_movies)

# Top 35 movies recommended for id 43
recs_35_for_43 = popular_recommendations('43', 35, ranked_movies)


def popular_recs_filtered(user_id, n_top, ranked_movies, years=None, genres=None):
    '''
    INPUT:
    user_id - the user_id (str) of the individual you are making recommendations for
    n_top - an integer of the number recommendations you want back
    ranked_movies - a pandas dataframe of the already ranked movies based on avg rating, count, and time
    years - a list of strings with years of movies
    genres - a list of strings with genres of movies

    OUTPUT:
    top_movies - a list of the n_top recommended movies by movie title in order best to worst
    '''
    # Filter movies based on year and genre
    if years is not None:
        ranked_movies = ranked_movies[ranked_movies['date'].isin(years)]

    if genres is not None:
        num_genre_match = ranked_movies[genres].sum(axis=1)
        ranked_movies = ranked_movies.loc[num_genre_match > 0, :]

    # create top movies list
    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies


# Top 20 movies recommended for id 1 with years=['2015', '2016', '2017', '2018'], genres=['History']
recs_20_for_1_filtered = popular_recs_filtered('1', 20, ranked_movies, years=['2015', '2016', '2017', '2018'], genres=['History'])

# Top 5 movies recommended for id 53968 with no genre filter but years=['2015', '2016', '2017', '2018']
recs_5_for_53968_filtered = popular_recs_filtered('53968', 5, ranked_movies, years=['2015', '2016', '2017', '2018'])

# Top 100 movies recommended for id 70000 with no year filter but genres=['History', 'News']
recs_100_for_70000_filtered = popular_recs_filtered('70000', 100, ranked_movies, genres=['History', 'News'])


