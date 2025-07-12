import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load ratings
df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_5\movie_rating.csv")

# Drop duplicate user and movie embeddings
user_emb = df[['user_emb_id']].drop_duplicates().reset_index(drop=True)
movie_emb = df[['movie_emb_id']].drop_duplicates().reset_index(drop=True)

# Dummy embedding vectors (you can replace with real vectors if available)
# We'll assume embedding ID itself is the index and use one-hot like sparse vectors for illustration
user_vectors = pd.get_dummies(user_emb['user_emb_id'])
movie_vectors = pd.get_dummies(movie_emb['movie_emb_id'])

# PCA Visualization
def plot_pca(data, label, title):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    plt.title(f"{title} (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.savefig(f"{label}_pca.png")
    plt.show()

# t-SNE Visualization
def plot_tsne(data, label, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(data)
    plt.figure(figsize=(6, 5))
    plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.6)
    plt.title(f"{title} (t-SNE)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.savefig(f"{label}_tsne.png")
    plt.show()

# Plot user and movie embeddings
plot_pca(user_vectors, "user", "User Embeddings")
plot_pca(movie_vectors, "movie", "Movie Embeddings")
plot_tsne(user_vectors, "user", "User Embeddings")
plot_tsne(movie_vectors, "movie", "Movie Embeddings")
