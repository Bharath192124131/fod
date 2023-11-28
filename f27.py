import numpy as np
from scipy import stats

# Assuming sample_data is a NumPy array containing the revenue data
sample_data = np.array([120, 150, 130, 160, 140, 170, 110, 180, 200, 190, ...])

# Specify the confidence level (e.g., 95%)
confidence_level = 0.95

# Calculate the mean and standard error of the mean
mean_revenue = np.mean(sample_data)
std_error = stats.sem(sample_data)

# Calculate the degrees of freedom
df = len(sample_data) - 1

# Calculate the t-value for the given confidence level and degrees of freedom
t_value = stats.t.ppf((1 + confidence_level) / 2, df)

# Calculate the margin of error
margin_of_error = t_value * std_error

# Calculate the confidence interval
confidence_interval = (mean_revenue - margin_of_error, mean_revenue + margin_of_error)

# Print the results
print(f"Sample Mean: {mean_revenue}")
print(f"Margin of Error: {margin_of_error}")
print(f"Confidence Interval ({confidence_level * 100}%): {confidence_interval}")
