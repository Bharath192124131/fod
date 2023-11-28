import pandas as pd
import numpy as np

# Load the temperature data from the CSV file
temperature_data = pd.read_csv('temperature_data.csv')

# Task 1: Calculate the mean temperature for each city
mean_temperatures = temperature_data.groupby('City')['Temperature'].mean()

# Task 2: Calculate the standard deviation of temperature for each city
std_dev_temperatures = temperature_data.groupby('City')['Temperature'].std()

# Task 3: Determine the city with the highest temperature range
temperature_range = temperature_data.groupby('City')['Temperature'].agg(lambda x: np.max(x) - np.min(x))
city_highest_range = temperature_range.idxmax()

# Task 4: Find the city with the most consistent temperature (lowest standard deviation)
city_most_consistent = std_dev_temperatures.idxmin()

# Display the results
print("Mean Temperatures:")
print(mean_temperatures)

print("\nStandard Deviation of Temperatures:")
print(std_dev_temperatures)

print(f"\nThe city with the highest temperature range is: {city_highest_range}")

print(f"The city with the most consistent temperature is: {city_most_consistent}")
