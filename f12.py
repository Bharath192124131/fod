import pandas as pd

# Replace this DataFrame with your actual sales data
sales_data = pd.DataFrame({
    'ProductID': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
    'QuantitySold': [10, 15, 8, 20, 12, 18, 25, 10, 22, 14, 12, 18, 30, 15, 20]
})

# Group by ProductID and sum the quantities sold for each product
product_sales = sales_data.groupby('ProductID')['QuantitySold'].sum()

# Sort the products based on total quantity sold in descending order
sorted_products = product_sales.sort_values(ascending=False)

# Get the top 5 products
top_5_products = sorted_products.head(5)

# Print the result
print("Top 5 Products Sold the Most:")
print(top_5_products)
