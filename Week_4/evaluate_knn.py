# knn_recommender_evaluation.py

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Step 1: Load Data
ratings_df = pd.read_csv(r'C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv')
movies_df = pd.read_csv(r'C:\Users\athar\coding\SOC-25\Week_4\test_data.csv')

# Step 2: Create User-Item Matrix
user_item_matrix = ratings_df.pivot_table(index='user_id', columns='movieId', values='rating', fill_value=0)
user_ids = user_item_matrix.index.tolist()
movie_ids = user_item_matrix.columns.tolist()
user_item_sparse = csr_matrix(user_item_matrix.values)

# Step 3: Fit k-NN Model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(user_item_sparse)

# Step 4: Recommendation Function
def recommend_movies_for_user(user_id, user_item_matrix, knn_model, user_ids, movie_ids, k=10, num_recommendations=50):
    if user_id not in user_ids:
        return []
    user_idx = user_ids.index(user_id)
    user_vector = user_item_matrix.iloc[user_idx].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(user_vector, n_neighbors=k+1)
    neighbor_indices = indices.flatten()[1:]  # Exclude the user themself

    neighbor_ratings = user_item_matrix.iloc[neighbor_indices]
    avg_ratings = neighbor_ratings.mean(axis=0)

    user_rated = set(user_item_matrix.columns[user_item_matrix.iloc[user_idx] > 0])

    recommendations = []
    for movie_id in avg_ratings.sort_values(ascending=False).index:
        if movie_id not in user_rated:
            recommendations.append(movie_id)
        if len(recommendations) == num_recommendations:
            break
    return recommendations

# Step 5: Evaluation Functions
def get_actual_liked_movies(ratings_df, user_id, min_rating=4):
    return set(ratings_df[(ratings_df['user_id'] == user_id) & (ratings_df['rating'] >= min_rating)]['movieId'])

def evaluate_knn_for_all_users(
    ratings_df, user_item_matrix, knn_model, user_ids, movie_ids,
    k=10, num_recommendations=10, max_users=50
):
    users = user_ids[:max_users]
    precisions = []
    recalls = []

    for i, user_id in enumerate(users):
        if i % 10 == 0:
            print(f"Evaluating user {i+1}/{len(users)} (user_id={user_id})...")
        # Get up to 50 recommendations, then filter to those the user has rated
        recommended_ids = recommend_movies_for_user(
            user_id, user_item_matrix, knn_model, user_ids, movie_ids, k, num_recommendations=50
        )
        user_rated_movies = set(ratings_df[ratings_df['user_id'] == user_id]['movieId'])
        # Only keep recommended movies that the user has rated
        rated_and_recommended = [mid for mid in recommended_ids if mid in user_rated_movies][:num_recommendations]
        actual_liked = get_actual_liked_movies(ratings_df, user_id)
        recommended = set(rated_and_recommended)
        true_positives = len(recommended & actual_liked)
        precision = true_positives / len(rated_and_recommended) if rated_and_recommended else 0
        recall = true_positives / len(actual_liked) if actual_liked else 0
        precisions.append(precision)
        recalls.append(recall)

        # Debug for first few users
        if i < 5:
            print(f"\nDEBUG for user {user_id}:")
            print("  Recommended movieIds (all):", recommended_ids)
            print("  Rated & recommended movieIds (for eval):", rated_and_recommended)
            print("  Actual liked movieIds:", actual_liked)
            print("  True positives:", recommended & actual_liked)

    mean_precision = sum(precisions) / len(precisions)
    mean_recall = sum(recalls) / len(recalls)
    print(f"\nMean Precision@{num_recommendations}: {mean_precision:.4f}")
    print(f"Mean Recall@{num_recommendations}: {mean_recall:.4f}")
    return precisions, recalls

# Step 6: Run Evaluation
if __name__ == "__main__":
    evaluate_knn_for_all_users(
        ratings_df, user_item_matrix, knn_model, user_ids, movie_ids,
        k=10, num_recommendations=10, max_users=50  # Increase max_users for full run
    )
