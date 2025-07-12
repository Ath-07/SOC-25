# 📦 Week 4: Classical Models & Evaluation (Midterm Submission)

This folder contains all work completed during **Week 4** of the ML Mentorship Program. The focus was on implementing **classical machine learning models** for recommendation, learning to evaluate them using relevant metrics, and reflecting on progress.

---

## 📚 Objectives of Week 4

- Implement classical ML models like **k-Nearest Neighbors (k-NN)** and **Logistic Regression**
- Evaluate model performance using:
  - **Precision@k**, **Recall@k** (for k-NN)
  - **Accuracy**, **Precision**, **Recall** (for Logistic Regression)
- Understand strengths/weaknesses of classical approaches in recommender systems

---

## 🧠 Learning Reflection (Week 4)

| Area                      | What I Knew Before     | What I Learned / Improved In          |
|---------------------------|------------------------|----------------------------------------|
| Evaluation Metrics        | Only Accuracy          | Learned how to implement and apply Precision@k, Recall@k |
| Recommender Systems       | Basic idea             | Built user-user collaborative filtering with k-NN |
| Binary Classification     | Knew theory            | Applied Logistic Regression on real data |
| Feature Engineering       | Little to none         | Used `user_emb_id` and `movie_emb_id` as model features |
| Model Comparison          | Not done before        | Compared two models using meaningful metrics |

---

## 🧪 Work Completed

### ✅ k-NN Recommender System
- Created **user-item matrix** from rating data
- Implemented **cosine similarity** to find nearest neighbors
- Generated **Top-5 recommendations** per user
- Evaluated model using **Precision@5** and **Recall@5**

### ✅ Logistic Regression Model
- Converted ratings to binary labels (`liked` = 1 if rating ≥ 4.0)
- Trained a **Logistic Regression** classifier using `user_emb_id`, `movie_emb_id`
- Evaluated using **accuracy**, **precision**, **recall**

### ✅ Model Comparison
| Metric         | k-NN Recommender     | Logistic Regression       |
|----------------|----------------------|----------------------------|
| Precision@5    | Evaluated            | N/A                        |
| Recall@5       | Evaluated            | N/A                        |
| Accuracy       | N/A                  | ✅ Achieved                |
| Precision      | N/A                  | ✅ Achieved                |
| Recall         | N/A                  | ✅ Achieved                |

---

## 📂 Folder Structure

week4_classical_models/
├── knn.py                # Implements user-based collaborative filtering using k-NN.
│                         # Generates top-N movie recommendations based on similar users.
│
├── logireg.py            # Applies Logistic Regression for binary classification of user preferences.
│                         # Uses movie/user embeddings to predict whether a user will like a movie.
│
├── movie_rating.csv      # Main dataset containing user ratings.
│                         # Includes: user_id, movieId, rating, timestamp, user_emb_id, movie_emb_id.
│
├── test_data.csv         # Metadata for movies.
│                         # Includes: movieId, title, genres.
│
├── README.md             # Documentation of Week 4 work.
│                         # Covers models, metrics, learnings, and project structure.
