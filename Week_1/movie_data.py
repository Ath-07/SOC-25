import pandas as pd

# Load dataset
dataset = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_1\test_data.csv")

# Extract unique genres
all_genres = set()
for genre_str in dataset['genres']:
    all_genres.update(genre_str.split('|'))
all_genres = sorted(list(all_genres))

# One-hot encode genres
def encode_genres(genre_str):
    genre_set = set(genre_str.split('|'))
    return [1 if genre in genre_set else 0 for genre in all_genres]

# Create dictionary: title -> genre vector
movie_data = {
    row['title']: encode_genres(row['genres']) 
    for _, row in dataset.iterrows()
}

print(movie_data)