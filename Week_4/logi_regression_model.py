import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from evaluation_utils import evaluate_all_users


# Load ratings data
ratings_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv")

# Step 1: Binarize ratings (liked = 1 if rating >= 4, else 0)
ratings_df['liked'] = ratings_df['rating'].apply(lambda x: 1 if x >= 4 else 0)

# Step 2: Keep only required columns
df_model = ratings_df[['user_id', 'movieId', 'liked']]

# Step 3: Encode user and movie IDs as categorical features
df_model['user_id'] = df_model['user_id'].astype('category').cat.codes
df_model['movieId'] = df_model['movieId'].astype('category').cat.codes

# Step 4: Create feature matrix (X) and target (y)
X = df_model[['user_id', 'movieId']]
y = df_model['liked']

# Step 5: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

# Step 6: Train the model
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = log_reg.predict(X_test)

# Step 8: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))


# Load original data
ratings_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv")
movies_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\test_data.csv")

# Create category encoding maps used earlier
user_map = {id_: i for i, id_ in enumerate(ratings_df['user_id'].astype('category').cat.categories)}
movie_map = {id_: i for i, id_ in enumerate(ratings_df['movieId'].astype('category').cat.categories)}
movie_reverse_map = {v: k for k, v in movie_map.items()}  # To reverse map to original IDs

# Convert movieId in movies_df to same format
movies_df['movieId_cat'] = movies_df['movieId'].map(movie_map)

def predict_for_user(user_original_id, K=5):
    if user_original_id not in user_map:
        print("User not found.")
        return

    user_id = user_map[user_original_id]

    # Movies already rated by the user
    rated_movies = ratings_df[ratings_df['user_id'] == user_original_id]['movieId'].map(movie_map).dropna().astype(int).tolist()

    # Candidate movies = all except rated
    all_movie_ids = set(movie_map.values())
    unrated_movies = list(all_movie_ids - set(rated_movies))

    # Create test samples (user_id, movie_id)
    test_samples = pd.DataFrame({'user_id': [user_id] * len(unrated_movies),
                                 'movieId': unrated_movies})

    # Predict probabilities
    probs = log_reg.predict_proba(test_samples)[:, 1]  # Prob of liked

    # Sort and get top-K
    top_indices = np.argsort(probs)[::-1][:K]
    top_movie_ids = [movie_reverse_map[test_samples.iloc[i]['movieId']] for i in top_indices]

    # Map to titles
    recommendations = movies_df[movies_df['movieId'].isin(top_movie_ids)][['movieId', 'title']]

    print(f"\nTop {K} recommendations for user {user_original_id}:\n")
    print(recommendations.reset_index(drop=True))

user_original_id = int(input("Enter user ID to get recommendations: "))
K = int(input("Enter number of recommendations (K): "))
predict_for_user(user_original_id, K)

# Evaluate average Precision@k and Recall@k
avg_prec, avg_rec = evaluate_all_users(ratings_df, k=5, user_map=user_map, movie_map=movie_map, movie_reverse_map=movie_reverse_map, log_reg=log_reg)

print(f"\nAverage Precision@5: {avg_prec:.4f}")
print(f"Average Recall@5: {avg_rec:.4f}")
