# knn_recommender_leave_one_out_debug.py

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# 1. Load data
ratings_df = pd.read_csv(r'C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv')
movies_df = pd.read_csv(r'C:\Users\athar\coding\SOC-25\Week_4\test_data.csv')

# 2. Split data: for each user, hold out one rating as test
train_list = []
test_list = []
for user_id, group in ratings_df.groupby('user_id'):
    if len(group) < 2:
        train_list.append(group)
        continue
    test_row = group.sample(1, random_state=42)
    train_rows = group.drop(test_row.index)
    train_list.append(train_rows)
    test_list.append(test_row)
train_df = pd.concat(train_list)
test_df = pd.concat(test_list)

# 3. Build user-item matrix from train data
user_item_matrix = train_df.pivot_table(index='user_id', columns='movieId', values='rating', fill_value=0)
user_ids = user_item_matrix.index.tolist()
movie_ids = user_item_matrix.columns.tolist()
user_item_sparse = csr_matrix(user_item_matrix.values)

# 4. Fit k-NN on train
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_item_sparse)

# 5. Recommend for each user and evaluate
k = 10
hits = 0
precisions = []
recalls = []

for i, user_id in enumerate(test_df['user_id'].unique()):
    if user_id not in user_ids:
        continue
    user_idx = user_ids.index(user_id)
    user_vector = user_item_matrix.iloc[user_idx].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=11)
    neighbor_indices = indices.flatten()[1:]  # Exclude self
    neighbor_ratings = user_item_matrix.iloc[neighbor_indices]
    avg_ratings = neighbor_ratings.mean(axis=0)
    user_rated = set(user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0])
    recommendations = [mid for mid in avg_ratings.sort_values(ascending=False).index if mid not in user_rated][:k]
    test_movies = set(test_df[test_df['user_id'] == user_id]['movieId'])
    relevant_recs = set(recommendations) & test_movies
    precision = len(relevant_recs) / k if k else 0
    recall = len(relevant_recs) / len(test_movies) if test_movies else 0
    hit = 1 if relevant_recs else 0
    precisions.append(precision)
    recalls.append(recall)
    hits += hit

    # Debug output for first 5 users
    if i < 5:
        print(f"\nDEBUG for user {user_id}:")
        print("  Recommended movieIds:", recommendations)
        print("  Test movieIds (held out):", list(test_movies))
        print("  Intersection (hit):", relevant_recs)
        if relevant_recs:
            print("  --> SUCCESS: At least one test movie was recommended.")
        else:
            print("  --> MISS: No test movies were recommended.")

print(f"\nMean Precision@{k}: {np.mean(precisions):.4f}")
print(f"Mean Recall@{k}: {np.mean(recalls):.4f}")
print(f"Hit Rate@{k}: {hits/len(precisions):.4f}")
