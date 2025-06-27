import pandas as pd

# Import your evaluation functions
from evaluation_utils import evaluate_all_users  # For logistic regression
from evaluate_knn import evaluate_knn_for_all_users  # For k-NN

# Load your data
ratings_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\movie_rating.csv")
movies_df = pd.read_csv(r"C:\Users\athar\coding\SOC-25\Week_4\test_data.csv")

# --- LOGISTIC REGRESSION EVALUATION ---
# You need to load your trained logistic regression model here.
# For illustration, we'll assume you have a function or code block that loads it:
import joblib
log_reg = joblib.load(r"C:\Users\athar\coding\SOC-25\Week_4\logi_regression_model.py")  # Replace with your actual model path

# Evaluate logistic regression
print("Evaluating Logistic Regression...")
lr_precisions, lr_recalls = evaluate_all_users(ratings_df, log_reg, k=10)
lr_precision = sum(lr_precisions) / len(lr_precisions)
lr_recall = sum(lr_recalls) / len(lr_recalls)

# --- k-NN EVALUATION ---
print("Evaluating k-NN...")
knn_precisions, knn_recalls = evaluate_knn_for_all_users(ratings_df, movies_df, k=10, num_recommendations=10)
knn_precision = sum(knn_precisions) / len(knn_precisions)
knn_recall = sum(knn_recalls) / len(knn_recalls)

# --- DISPLAY RESULTS ---
results = {
    "Model": ["Logistic Regression", "k-NN"],
    "Precision@10": [lr_precision, knn_precision],
    "Recall@10": [lr_recall, knn_recall]
}

results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)
