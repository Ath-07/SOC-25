import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
ratings_df = pd.read_csv(r"C:\Users\athar\coding\test\movie_rating.csv")
movies_df = pd.read_csv(r"C:\Users\athar\coding\test\test_data.csv")

# Label ratings: like (1) or not like (0)
ratings_df['liked'] = (ratings_df['rating'] >= 4.0).astype(int)

# Select features: we'll use user_emb_id and movie_emb_id
X = ratings_df[['user_emb_id', 'movie_emb_id']]
y = ratings_df['liked']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
