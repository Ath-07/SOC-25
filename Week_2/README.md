# Week 2: Probability & Exploratory Data Analysis (EDA)

## Objectives
- Understand basic probability and statistical concepts.
- Analyze movie ratings data to uncover insights.
- Visualize user-item interaction patterns and rating behavior.

## Files

- **eda.py**  
  Performs the following tasks:
  - Loads `ratings.csv` and `movies.csv`.
  - Merges the datasets using `movieId`.
  - Computes rating statistics and probability distribution.
  - Analyzes rating frequency and user activity.
  - Visualizes:
    - Probability distribution of ratings
    - Top 20 most-rated movies
    - Top movies by average rating (min 50 ratings)
    - Ratings per user

## Key Analyses

- **Rating Probability Distribution**  
  Shows how often each rating (0.5 to 5.0) occurs in the dataset.

- **User Activity Analysis**  
  Histogram of how many movies users have rated.

- **Movie Popularity**  
  Top 10 most frequently rated movies.

- **Top Rated Movies**  
  Movies with the highest average ratings (minimum 50 ratings for reliability).

## How to Run

Make sure `ratings.csv` and `movies.csv` are in the same folder as `eda.py`. Then run:

```bash
python eda.py
