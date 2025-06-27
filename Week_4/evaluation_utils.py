import pandas as pd
import numpy as np

def evaluate_user_precision_recall(user_original_id, k, user_map, movie_map, movie_reverse_map, ratings_df, log_reg):
    if user_original_id not in user_map:
        return None, None

    user_id = user_map[user_original_id]

    rated_movies = ratings_df[ratings_df['user_id'] == user_original_id]['movieId'].map(movie_map).dropna().astype(int).tolist()
    all_movie_ids = set(movie_map.values())
    unrated_movies = list(all_movie_ids - set(rated_movies))

    if not unrated_movies:
        return None, None

    test_samples = pd.DataFrame({
        'user_id': [user_id] * len(unrated_movies),
        'movieId': unrated_movies
    })

    probs = log_reg.predict_proba(test_samples)[:, 1]
    top_indices = np.argsort(probs)[::-1][:k]
    top_movie_ids = [test_samples.iloc[i]['movieId'] for i in top_indices]
    original_top_ids = [movie_reverse_map[mid] for mid in top_movie_ids]

    liked_movies = ratings_df[(ratings_df['user_id'] == user_original_id) & (ratings_df['liked'] == 1)]['movieId'].tolist()

    true_positives = len(set(original_top_ids) & set(liked_movies))
    precision = true_positives / k
    recall = true_positives / len(liked_movies) if liked_movies else 0

    return precision, recall


def evaluate_all_users(ratings_df, k, user_map, movie_map, movie_reverse_map, log_reg):
    user_ids = ratings_df['user_id'].unique()
    precision_scores = []
    recall_scores = []

    for uid in user_ids:
        prec, rec = evaluate_user_precision_recall(uid, k, user_map, movie_map, movie_reverse_map, ratings_df, log_reg)
        if prec is not None:
            precision_scores.append(prec)
            recall_scores.append(rec)

    return np.mean(precision_scores), np.mean(recall_scores)
