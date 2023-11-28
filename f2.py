import pandas as pd

# Assuming 'order_data' is your DataFrame

# 1. The total number of orders made by each customer
total_orders_by_customer = order_data.groupby('customer_id')['order_date'].count()

# 2. The average order quantity for each product
average_order_quantity_per_product = order_data.groupby('product_name')['order_quantity'].mean()

# 3. The earliest and latest order dates in the dataset
earliest_order_date = order_data['order_date'].min()
latest_order_date = order_data['order_date'].max()

# Print the results
print("Total number of orders by each customer:")
print(total_orders_by_customer)

print("\nAverage order quantity for each product:")
print(average_order_quantity_per_product)

print("\nEarliest order date:", earliest_order_date)
print("Latest order date:", latest_order_date)
