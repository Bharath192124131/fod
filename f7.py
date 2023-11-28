import numpy as np

# Assuming house_data is a NumPy array with columns: [bedrooms, square_footage, sale_price]
# Replace this array with your actual data
house_data = np.array([
    [3, 1500, 250000],
    [4, 1800, 300000],
    [5, 2000, 350000],
    [3, 1600, 280000],
    [6, 2200, 400000],
    [4, 1700, 320000],
])

# Find indices of houses with more than four bedrooms
more_than_four_bedrooms_indices = house_data[:, 0] > 4

# Filter house_data to include only houses with more than four bedrooms
houses_more_than_four_bedrooms = house_data[more_than_four_bedrooms_indices]

# Calculate the average sale price of houses with more than four bedrooms
average_sale_price_more_than_four_bedrooms = np.mean(houses_more_than_four_bedrooms[:, -1])

# Print the result
print("Average Sale Price of Houses with More Than Four Bedrooms:", average_sale_price_more_than_four_bedrooms)
