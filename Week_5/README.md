# 📚 Week 5: Hybrid Models & Cold-Start Problem

## ✅ Objectives

- Develop a **hybrid recommendation system** that combines collaborative filtering and content-based filtering.
- Address the **cold-start problem** using item metadata (genres).
- Visualize **user and item embeddings** using dimensionality reduction techniques like PCA and t-SNE.

---

## 🛠️ What I Did

This week focused on deepening my understanding of hybrid recommender systems and handling limitations of traditional models. Here’s what I built and learned:

### 1. 🔁 Hybrid Recommendation System (`hybrid_model.py`)
- Combined user-based collaborative filtering (using **k-NN** on user-item matrix) and content-based filtering (based on **genre similarity**).
- Used cosine similarity to compute both user and movie similarities.
- Designed the system to be **interactive** – users can input a `user_emb_id` to get recommendations.
- Weighted final recommendations from both approaches (collaborative and content-based).

### 2. 🧊 Cold-Start Handling (`cold_start.py`)
- For **new users**, the system prompts genre preferences (e.g., `Action|Comedy`) and recommends movies using genre similarity.
- For **new movies** (not yet rated), genre metadata helps estimate similarity to existing items.
- Used `CountVectorizer` to encode genres and `cosine_similarity` to compute similarity vectors.

### 3. 📊 Embedding Visualization (`visualize_embeddings.py`)
- Visualized both **user** and **movie** embeddings using:
  - **PCA** – for linear dimensionality reduction.
  - **t-SNE** – to capture non-linear clustering structures.
- Generated clear scatter plots to show user and item grouping patterns in 2D space.

---

## 💡 What I Learned

| Concept | Improvement |
|--------|-------------|
| Hybrid filtering | Learned how to combine collaborative and content-based scores using normalization |
| Cold-start problem | Understood how metadata helps in absence of user/item history |
| Embedding visualization | Practiced applying PCA & t-SNE for intuitive 2D plotting |
| Code design | Made scripts interactive by asking for real-time input from user |

---

## 🗂️ Folder Structure

