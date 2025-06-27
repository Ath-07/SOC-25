# 📊 Week 3 — Matrix Factorization & Evaluation

This week focuses on applying **matrix factorization** for collaborative filtering in a movie recommendation engine using **Singular Value Decomposition (SVD)**. The goal is to predict user ratings and evaluate the model's performance.

---

## ✅ Components

### `svd_recommender.py`
- Loads user-item ratings from `ratings.csv`.
- Uses the `Surprise` library's SVD algorithm for collaborative filtering.
- Trains on 80% of the data, tests on the remaining 20%.
- Evaluates predictions using **RMSE** (Root Mean Square Error).

---

## 📚 Key Concepts

### 🔸 Matrix Factorization (SVD)
SVD factorizes the user-item rating matrix \( R \) as:

\[
R \approx U \Sigma V^T
\]

Where:
- \( U \): User latent feature matrix
- \( \Sigma \): Importance of each latent feature
- \( V^T \): Item latent feature matrix

This helps estimate missing ratings by projecting users and items into a shared feature space.

### 🔸 RMSE Evaluation
To assess accuracy:

\[
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n}(r_i - \hat{r}_i)^2}
\]

Lower RMSE means better predictive performance.

---

## 📝 Week 3 Checklist

- [x] Completed study of SVD.
- [x] Implemented collaborative filtering using SVD.
- [x] Evaluated model performance using RMSE.

---

## 📂 Files in this Folder

- `svd_recommender.py`: Implements the SVD-based recommendation system.
- `ratings.csv`: Contains user-item ratings for training and evaluation.