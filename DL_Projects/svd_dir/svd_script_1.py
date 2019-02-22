import pandas as pd
from scipy import matrix
import blaze as bz
import numpy as np
import matplotlib.pyplot as plt


movies = pd.read_csv('/home/wasi/Desktop/main_df/new_movies.csv')
reviews = pd.read_csv('/home/wasi/Desktop/main_df/new_reviews.csv')


def create_train_test(reviews, order_by, training_size, testing_size):
    reviews_new = reviews.sort_values(order_by)
    training_df = reviews_new.head(training_size)
    validation_df = reviews_new.iloc[training_size:training_size+testing_size]

    return training_df, validation_df


train_df, val_df = create_train_test(reviews, 'date', 8000, 2000)

# train_df.to_csv("/home/wasi/Desktop/main_df/train_df.csv", index=False)
# val_df.to_csv("/home/wasi/Desktop/main_df/train_df.csv", index=False)

print(train_df.shape, val_df.shape)


def FunkSVD(ratings_mat, latent_features=4, learning_rate=0.0001, iters=100):
    n_users = ratings_mat.shape[0]
    n_movies = ratings_mat.shape[1]
    num_ratings = np.count_nonzero(~np.isnan(ratings_mat))

    user_mat = np.random.rand(n_users, latent_features)
    movie_mat = np.random.rand(latent_features, n_movies)

    sse_accum = 0

    print("Optimizaiton Statistics")
    print("Iterations | Mean Squared Error ")

    for iteration in range(iters):

        old_sse = sse_accum
        sse_accum = 0

        for i in range(n_users):
            for j in range(n_movies):

                if ratings_mat[i, j] > 0:

                    diff = ratings_mat[i, j] - np.dot(user_mat[i, :], movie_mat[:, j])

                    sse_accum += diff**2

                    for k in range(latent_features):
                        user_mat[i, k] += learning_rate * (2*diff*movie_mat[k, j])
                        movie_mat[k, j] += learning_rate * (2*diff*user_mat[i, k])

        # print results
        print("%d \t\t %f" % (iteration+1, sse_accum / num_ratings))

    return user_mat, movie_mat


train_user_item = train_df[['user_id', 'movie_id', 'rating', 'timestamp']]
train_data_df = train_user_item.groupby(['user_id', 'movie_id'])['rating'].max().unstack()
train_data_np = np.array(train_data_df)

user_mat, movie_mat = FunkSVD(train_data_np, latent_features=4, learning_rate=0.005, iters=500)


def predict_rating(user_matrix, movie_matrix, user_id, movie_id):
    user_ids_series = np.array(train_data_df.index)
    movie_ids_series = np.array(train_data_df.columns)

    user_row = np.where(user_ids_series == user_id)[0][0]
    movie_col = np.where(movie_ids_series == movie_id)[0][0]

    pred = np.dot(user_matrix[user_row, :], movie_matrix[:, movie_col])

    return pred


pred_val = predict_rating(user_mat, movie_mat, 8854, 302886)
print(pred_val)


def print_prediction_summary(user_id, movie_id, prediction):
    movie_name = str(movies[movies['movie_id'] == movie_id]['movie'])[5:]
    movie_name = movie_name.replace('\nName: movie, dtype: object', '')
    print("For user {} we predict a {} rating for the movie {}.".format(user_id, round(prediction, 2),
                                                                        str(movie_name)))


print_prediction_summary(8854, 302886, pred_val)


def validation_comparison(val_df, num_preds):
    val_users = np.array(val_df['user_id'])
    val_movies = np.array(val_df['movie_id'])
    val_ratings = np.array(val_df['rating'])

    for idx in range(num_preds):
        pred = predict_rating(user_mat, movie_mat, val_users[idx], val_movies[idx])
        print("The actual rating for user {} on movie {} is {}.\n While the predicted rating is {}.".
              format(val_users[idx], val_movies[idx], val_ratings[idx], round(pred)))


validation_comparison(val_df, 6)


