import numpy as np

# Replace this array with your actual sales data
sales_data = np.array([
    [100],
    [150],
    [120],
    [200],
    # ... additional rows ...
])

# Extract the column containing product prices
product_prices = sales_data[:, 0]

# Calculate the average price
average_price = np.mean(product_prices)

# Print the result
print("Average Price of Products Sold in the Past Month:", average_price)
