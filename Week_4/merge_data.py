import pandas as pd

def load_and_merge_data(ratings_path, movies_path):
    # Load both CSVs
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)

    # Merge them on movieId to get titles with ratings
    merged_df = pd.merge(ratings, movies, on="movieId")

    return merged_df

def create_user_item_matrix(df):
    """
    Create a matrix where rows = users, columns = movies, values = ratings.
    If a user hasn't rated a movie, it's left as 0.
    """
    user_item_matrix = df.pivot_table(index='user_id', columns='movieId', values='rating')
    user_item_matrix = user_item_matrix.fillna(0)  # fill unrated with 0

    return user_item_matrix

# Paths to CSVs
ratings_path = r"C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv"
movies_path = r"C:\Users\athar\coding\SOC-25\Week_4\test_data.csv"

# Load & merge
merged_df = load_and_merge_data(ratings_path, movies_path)

# Create matrix
user_item_matrix = create_user_item_matrix(merged_df)

print("User-Item Matrix created with shape:", user_item_matrix.shape)
user_item_matrix.to_csv('user_item_matrix.csv')
print("Matrix saved as 'user_item_matrix.csv'")
print("First few rows of the User-Item Matrix:")
print(user_item_matrix.head())