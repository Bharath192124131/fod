import numpy as np

# Replace this array with your actual fuel efficiency data
fuel_efficiency = np.array([25, 30, 22, 28, 35, 18, 26, 32, 30, 24])

# Calculate the average fuel efficiency
average_fuel_efficiency = np.mean(fuel_efficiency)

# Print the result
print("Average Fuel Efficiency:", average_fuel_efficiency, "miles per gallon")

# Replace these indices with the indices of the car models you want to compare
model1_index = 2
model2_index = 5

# Get fuel efficiency values for the specified car models
fuel_efficiency_model1 = fuel_efficiency[model1_index]
fuel_efficiency_model2 = fuel_efficiency[model2_index]

# Calculate the percentage improvement
percentage_improvement = ((fuel_efficiency_model2 - fuel_efficiency_model1) / fuel_efficiency_model1) * 100

# Print the result
print("Percentage Improvement in Fuel Efficiency:", percentage_improvement, "%")
