import pandas as pd

# Load the dataset
ratings_df = pd.read_csv(r"C:\Users\athar\coding\test\movie_rating.csv")

# Create a user-item matrix
user_item_matrix = ratings_df.pivot_table(index='user_emb_id', columns='movie_emb_id', values='rating').fillna(0)
#print(user_item_matrix)

from sklearn.neighbors import NearestNeighbors

# Fit k-NN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

# Choose user index (you can input your own later)
user_index = 0

# Get similar users
distances, indices = knn.kneighbors([user_item_matrix.iloc[user_index]], n_neighbors=6)

# Collect movies from similar users
similar_users = user_item_matrix.iloc[indices.flatten()[1:]]
mean_ratings = similar_users.mean(axis=0)

# Recommend top N movies the target user hasn't seen
user_ratings = user_item_matrix.iloc[user_index]
unrated_movies = user_ratings[user_ratings == 0]
recommendations = mean_ratings[unrated_movies.index].sort_values(ascending=False).head(5)

# Output recommended movie IDs
recommended_movie_ids = recommendations.index.tolist()
print(recommended_movie_ids)

movies_df = pd.read_csv(r"C:\Users\athar\coding\test\test_data.csv")

# Map movieId to title
movie_id_to_title = dict(zip(movies_df['movieId'], movies_df['title']))

# Print movie titles for recommendations
print("\nTop 5 recommended movies:")
for movie_id in recommended_movie_ids:
    print(movie_id_to_title.get(movie_id, f"Movie {movie_id}"))

def precision_at_k(recommended, relevant, k):
    recommended_at_k = recommended[:k]
    relevant_set = set(relevant)
    hits = sum(1 for movie in recommended_at_k if movie in relevant_set)
    return hits / k

def recall_at_k(recommended, relevant, k):
    relevant_set = set(relevant)
    hits = sum(1 for movie in recommended[:k] if movie in relevant_set)
    return hits / len(relevant) if relevant else 0

# ----- Evaluation for Multiple Users -----
k = 5
num_users_to_test = 20  # You can increase this

precisions = []
recalls = []

for user_index in range(num_users_to_test):
    try:
        # Get current user and recommendations
        user_vector = user_item_matrix.iloc[user_index]
        user_id = user_item_matrix.index[user_index]

        distances, indices = knn.kneighbors([user_vector], n_neighbors=6)
        similar_users = user_item_matrix.iloc[indices.flatten()[1:]]
        mean_ratings = similar_users.mean(axis=0)

        user_ratings = user_vector
        unrated_movies = user_ratings[user_ratings == 0]
        recommendations = mean_ratings[unrated_movies.index].sort_values(ascending=False).head(k)
        recommended_ids = recommendations.index.tolist()

        # Get relevant movies for the user (ground truth)
        relevant_ids = ratings_df[
            (ratings_df['user_emb_id'] == user_id) & (ratings_df['rating'] >= 4.0)
        ]['movie_emb_id'].tolist()

        if not relevant_ids:
            continue  # Skip users with no relevant ground truth

        p = precision_at_k(recommended_ids, relevant_ids, k)
        r = recall_at_k(recommended_ids, relevant_ids, k)

        precisions.append(p)
        recalls.append(r)

    except Exception as e:
        print(f"Error at user {user_index}: {e}")
        continue

# ----- Results -----
print(f"\nAverage Precision@{k}: {sum(precisions)/len(precisions):.4f}")
print(f"Average Recall@{k}: {sum(recalls)/len(recalls):.4f}")
