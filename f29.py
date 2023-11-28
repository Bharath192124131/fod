import pandas as pd
import numpy as np
from scipy import stats

# Load customer reviews dataset (replace 'customer_reviews.csv' with your actual file path)
reviews_df = pd.read_csv('customer_reviews.csv')

# Assuming the dataset has a column 'rating' representing customer ratings
ratings = reviews_df['rating']

# Calculate mean and standard error of the mean
mean_rating = np.mean(ratings)
std_error = stats.sem(ratings)

# Specify the confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the margin of error
margin_of_error = stats.t.ppf((1 + confidence_level) / 2, len(ratings) - 1) * std_error

# Calculate confidence interval
confidence_interval = (mean_rating - margin_of_error, mean_rating + margin_of_error)

# Display results
print(f"Average Rating: {mean_rating:.2f}")
print(f"Margin of Error: {margin_of_error:.2f}")
print(f"Confidence Interval ({confidence_level * 100}%): {confidence_interval}")
