import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer

# Load data
ratings_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_5\movie_rating.csv")
movies_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_5\test_data.csv")

# Create user-item matrix
user_item_matrix = ratings_df.pivot_table(index='user_emb_id', columns='movie_emb_id', values='rating').fillna(0)

# Genre similarity matrix
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = vectorizer.fit_transform(movies_df['genres'])
genre_sim = cosine_similarity(genre_matrix)

# Fit k-NN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(user_item_matrix)

# 🧑 Ask user for input
available_users = list(user_item_matrix.index)
print(f"Available user_emb_ids: {available_users[:10]} ...")
user_input = input("Enter a user_emb_id from above: ")

if not user_input.isdigit() or int(user_input) not in available_users:
    print("Invalid user_emb_id.")
    exit()

user_index = int(user_input)

# Find similar users
distances, indices = knn.kneighbors([user_item_matrix.loc[user_index]], n_neighbors=6)
similar_users = user_item_matrix.iloc[indices.flatten()[1:]]

# Collaborative score
mean_ratings = similar_users.mean()

# Content-based score
genre_df = movies_df.copy()
genre_df.index = genre_df['movieId']
movie_id_to_emb = ratings_df.drop_duplicates('movieId')[['movieId', 'movie_emb_id']].set_index('movieId').to_dict()['movie_emb_id']
genre_df['movie_emb_id'] = genre_df['movieId'].map(movie_id_to_emb)
genre_df = genre_df.dropna(subset=['movie_emb_id'])

user_rated = user_item_matrix.loc[user_index]
unrated_movies = user_rated[user_rated == 0]

collab_score = mean_ratings[unrated_movies.index]
genre_score = pd.Series(index=unrated_movies.index, dtype=float)

for movie in unrated_movies.index:
    try:
        genre_idx = genre_df[genre_df['movie_emb_id'] == movie].index[0]
        similarity_vector = genre_sim[genre_idx]
        genre_score[movie] = similarity_vector.mean()
    except:
        genre_score[movie] = 0

# Normalize & combine
collab_score.fillna(0, inplace=True)
genre_score.fillna(0, inplace=True)

final_score = 0.5 * collab_score + 0.5 * genre_score
recommendations = final_score.sort_values(ascending=False).head(5)

# Map IDs to titles
movie_map = dict(zip(movies_df['movieId'], movies_df['title']))
id_map = dict(zip(ratings_df['movieId'], ratings_df['movie_emb_id']))

print("\nTop 5 Hybrid Recommendations:")
for movie_id in recommendations.index:
    title = movie_map.get(movie_id, f"Movie {movie_id}")
    print(f"- {title}")

