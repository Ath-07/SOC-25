# Week 4: Classical Models & k-NN Evaluation

## Objectives
- Understand and implement user-based k-Nearest Neighbors (k-NN) for collaborative filtering.
- Prepare and split data for offline evaluation.
- Evaluate recommendations using Precision@k, Recall@k, and Hit Rate@k.
- Debug and interpret recommender performance.

---

## Tasks Completed
- ✅ Studied k-NN collaborative filtering and evaluation metrics.
- ✅ Preprocessed the MovieLens dataset (`ratings.csv` and `movies.csv`).
- ✅ Implemented user-based k-NN using `sklearn.neighbors.NearestNeighbors`.
- ✅ Split data into training and test sets for each user.
- ✅ Evaluated recommendations with Precision@k, Recall@k, and Hit Rate@k.
- ✅ Added debug output to inspect recommendations and test hits.

---

## Dataset
- **Source:** Provided `ratings.csv` (userId, movieId, rating, timestamp) and `movies.csv` (movieId, title, genres).
- Used only userId, movieId, and rating for collaborative filtering.
- Data is split per user into training and test sets.

---

## k-NN Recommender

- Built a user-item matrix (users as rows, movies as columns, ratings as values).
- Fitted a k-NN model (`NearestNeighbors` with cosine similarity) on the training matrix.
- For each user, found k most similar users (neighbors).
- Recommended top-N movies that neighbors liked but the user has not rated in training.

---

## Evaluation

- For each user, compared recommendations to their test set ratings.
- **Metrics:**
  - **Precision@k:** Fraction of top-k recommendations matching the test set.
  - **Recall@k:** Fraction of test set movies recovered.
  - **Hit Rate@k:** Percentage of users for whom at least one test movie was recommended.
- Printed debug info for the first five users (recommended movie IDs, test movies, intersection).

---