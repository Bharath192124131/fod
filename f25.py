import numpy as np

# Assuming data is a NumPy array containing the dataset
data = np.array([23, 45, 67, 34, 56, 78, 90, 12, 43, 65])

# Mean estimation
mean_estimate = np.mean(data)

# Variance estimation
variance_estimate = np.var(data)

# Simple random sampling example (randomly selecting 5 data points)
sample_size = 5
random_sample = np.random.choice(data, size=sample_size, replace=False)

# Print the estimates and the random sample
print(f"Mean Estimate: {mean_estimate}")
print(f"Variance Estimate: {variance_estimate}")
print(f"Random Sample: {random_sample}")
