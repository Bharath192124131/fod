import matplotlib.pyplot as plt
import numpy as np

# Assuming temperature and rainfall are NumPy arrays containing the daily temperature and rainfall data
temperature = np.array([25, 28, 30, 20, 22, 25, 27, 29, 26, 24])
rainfall = np.array([5, 8, 2, 12, 10, 6, 4, 9, 7, 11])

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(temperature, rainfall)[0, 1]

# Create a scatter plot
plt.scatter(temperature, rainfall)
plt.title("Scatter Plot of Temperature vs. Rainfall")
plt.xlabel("Temperature (°C)")
plt.ylabel("Rainfall (mm)")
plt.show()

# Print the correlation coefficient
print(f"Correlation Coefficient: {correlation_coefficient}")
