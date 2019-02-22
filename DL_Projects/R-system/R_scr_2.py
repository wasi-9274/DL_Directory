import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

movies = pd.read_csv("/home/wasi/Desktop/main_df/new_movies.csv")
reviews = pd.read_csv("/home/wasi/Desktop/main_df/new_reviews.csv")
print(movies.head())
print(reviews.head())


def create_ranked_df(movies, reviews):

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
    top_movies = list(ranked_movies['movie'][:n_top])
    return top_movies


# Top 20 movies recommended for id 1

ranked_movies = create_ranked_df(movies, reviews)
recs_20_for_1 = popular_recommendations('1', 20, ranked_movies)

for m in recs_20_for_1:
    print("Movies recommended: ", m)

print("=" * 60)


def popular_recs_filtered(user_id, n_top, ranked_movies, years=None, genres=None):
    # Filter movies based on year and genre
    if years is not None:
        ranked_movies = ranked_movies[ranked_movies['date'].isin(years)]

    if genres is not None:
        num_genre_match = ranked_movies[genres].sum(axis=1)
        ranked_movies = ranked_movies.loc[num_genre_match > 0, :]

    # create top movies list
    top_movies = list(ranked_movies['movie'][:n_top])

    return top_movies


recs_20_for_1_filtered = popular_recs_filtered('1', 20, ranked_movies, years=['2015', '2016', '2017', '2018'],
                                               genres=['History'])
for m in recs_20_for_1:
    print("Movies recommended from adding two seperate arguments : ", m)
