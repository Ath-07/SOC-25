# Week 1: Linear Algebra & Content-Based Filtering

## Objectives
- Understand vector operations and cosine similarity.
- Implement cosine similarity from scratch.
- Build a basic content-based movie recommender using genre metadata.

## Tasks Completed
- ✅ Studied Linear Algebra fundamentals (vectors, dot product, magnitude, angle between vectors).
- ✅ Implemented cosine similarity function manually in `cosine_similarity.py`.
- ✅ Collected and preprocessed the dataset (`test_data.csv`).
- ✅ Built a content-based recommender in `recommender.py` using genre-based vector representations.

## Dataset
- Source: Provided `test_data.csv` containing columns: `movieId`, `title`, and `genres`.
- Genres are represented as pipe-separated (`|`) strings (e.g., `Comedy|Romance`).
- Used one-hot encoding to convert genres into numerical feature vectors.

## Cosine Similarity
- Implemented manually in `cosine_similarity.py` without using external libraries like `sklearn`.  
- Edge case handled: returns `0.0` if either vector has zero magnitude.

## Data processing
- `movie_data.py` loads data from `test_data.csv` using pandas.
- Processes it to give value `1` to genre it has and `0` to genre it does not in an array.
- Stores movie name along with above array as dictionary `movie_data`.

## Recommender System
Implemented in `recommender.py`. It uses:
- The `cosine_similarity` function to calculate similarity between movie vectors.
- The processed movie data from `movie_data.py`.

### Usage
```python
recommend("Toy Story (1995)", top_n=3)

