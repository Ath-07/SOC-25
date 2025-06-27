import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class KNNRecommender:
    def __init__(self, ratings_df, movies_df, k=10):
        self.k = k
        self.ratings_df = ratings_df
        self.movies_df = movies_df
        self.user_item_matrix, self.user_ids, self.movie_ids = self._create_user_item_matrix()
        self.model = self._fit_knn()

        # Create mappings for easy lookup
        self.movieId_to_title = dict(zip(movies_df['movieId'], movies_df['title']))
        self.title_to_movieId = dict(zip(movies_df['title'], movies_df['movieId']))
        self.userId_to_index = {uid: idx for idx, uid in enumerate(self.user_ids)}
        self.index_to_userId = {idx: uid for idx, uid in enumerate(self.user_ids)}

    def _create_user_item_matrix(self):
        user_item = self.ratings_df.pivot_table(index='user_id', columns='movieId', values='rating', fill_value=0)
        user_ids = user_item.index.tolist()
        movie_ids = user_item.columns.tolist()
        return csr_matrix(user_item.values), user_ids, movie_ids

    def _fit_knn(self):
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(self.user_item_matrix)
        return model

    def recommend_movies(self, user_id, num_recommendations=10, min_rating=4.0):
        if user_id not in self.userId_to_index:
            print(f"User {user_id} not found.")
            return []

        user_idx = self.userId_to_index[user_id]
        user_vector = self.user_item_matrix[user_idx]

        # Find k nearest neighbors (excluding the user themself)
        distances, indices = self.model.kneighbors(user_vector, n_neighbors=self.k + 1)
        neighbor_indices = indices.flatten()[1:]  # skip the first (the user themself)

        # Aggregate ratings from neighbors
        neighbor_ratings = self.user_item_matrix[neighbor_indices].toarray()
        avg_ratings = neighbor_ratings.mean(axis=0)

        # Get movies the user has already rated
        user_rated = set(np.where(user_vector.toarray().flatten() > 0)[0])

        # Recommend movies not yet rated by the user, sorted by neighbors' average rating
        recommendations = []
        for idx in np.argsort(avg_ratings)[::-1]:
            if idx not in user_rated:
                movie_id = self.movie_ids[idx]
                title = self.movieId_to_title.get(movie_id, "Unknown")
                recommendations.append(title)
                if len(recommendations) == num_recommendations:
                    break
        return recommendations

# Example usage:
if __name__ == "__main__":
    ratings_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv")
    movies_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\test_data.csv")
    recommender = KNNRecommender(ratings_df, movies_df, k=10)
    
    # Prompt user for user ID input
    user_id_input = input("Enter user ID to get recommendations: ")
    try:
        user_id = int(user_id_input)
        recs = recommender.recommend_movies(user_id, num_recommendations=10)
        if recs:
            print(f"\nRecommendations for user {user_id}:")
            for title in recs:
                print(title)
        else:
            print(f"No recommendations found for user {user_id}.")
    except ValueError:
        print("Invalid input. Please enter a valid integer user ID.")

