import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Load movie metadata
movies_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_5\test_data.csv")
movies_df['genres'] = movies_df['genres'].fillna('')

# Use CountVectorizer to encode genres as vectors
vectorizer = CountVectorizer(tokenizer=lambda x: x.split('|'))
genre_matrix = vectorizer.fit_transform(movies_df['genres'])

# Compute pairwise cosine similarity between movies
genre_similarity = cosine_similarity(genre_matrix)

# Create a mapping from movieId to index
movieId_to_index = dict(zip(movies_df['movieId'], movies_df.index))
index_to_title = dict(zip(movies_df.index, movies_df['title']))

# === Cold-Start Movie ===
def recommend_for_new_movie(input_movie_id, top_n=5):
    idx = movieId_to_index.get(input_movie_id)
    if idx is None:
        print("❌ Movie not found.")
        return
    sim_scores = list(enumerate(genre_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    print(f"\n📽️ Recommendations for new movie (ID: {input_movie_id}):")
    for i, score in sim_scores:
        print(f"- {index_to_title[i]} (similarity: {score:.2f})")

# === Cold-Start User ===
def recommend_for_new_user(preferred_genres, top_n=5):
    # Convert preferred genres to genre vector
    user_vector = vectorizer.transform([preferred_genres])
    similarity_scores = cosine_similarity(user_vector, genre_matrix).flatten()
    
    top_indices = similarity_scores.argsort()[::-1][:top_n]
    print(f"\n👤 Recommendations for new user with genres: {preferred_genres}")
    for i in top_indices:
        print(f"- {index_to_title[i]} (similarity: {similarity_scores[i]:.2f})")

# Prompt user for preferred genres
user_input = input("Enter preferred genres (e.g., Action|Comedy): ")

# Transform genre into vector
user_vector = vectorizer.transform([user_input])

# Compute similarity with movies
similarity_scores = cosine_similarity(user_vector, genre_matrix).flatten()
top_indices = similarity_scores.argsort()[::-1][:5]

print("\nRecommended movies based on your preferences:")
for idx in top_indices:
    print(f"- {movies_df.iloc[idx]['title']}")
