# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
ratings = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_2\movie_rating.csv")
print(ratings.head())
movies = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_2\test_data.csv")
print(movies.head())

# Merge datasets on movieId
data = pd.merge(ratings, movies, on="movieId")
print(data.head())


# 1. Distribution of ratings
plt.figure(figsize=(8, 5))
sns.histplot(ratings['rating'], bins=9, kde=True) # type: ignore
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Number of ratings per movie
rating_counts = data.groupby('title').size().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
rating_counts.head(20).plot(kind='bar')
plt.title('Top 20 Most Rated Movies')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

# 3. Average rating per movie
avg_ratings = data.groupby('title')['rating'].mean().sort_values(ascending=False)

plt.figure(figsize=(10, 6))
avg_ratings.head(20).plot(kind='bar')
plt.title('Top 20 Highest Rated Movies (avg)')
plt.ylabel('Average Rating')
plt.xticks(rotation=75)
plt.tight_layout()
plt.show()

rating_probs = ratings['rating'].value_counts(normalize=True).sort_index()
print("\n--- Probability of Each Rating ---")
print(rating_probs)

sns.barplot(x=rating_probs.index, y=rating_probs.values)
plt.title('Probability of Each Rating')
plt.xlabel('Rating') 
plt.show()   