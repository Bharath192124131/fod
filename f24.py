from collections import Counter

# Assuming product_data is a dictionary where keys are product names and values are the number of times each product was sold
product_data = {'ProductA': 50, 'ProductB': 30, 'ProductC': 70, 'ProductD': 40}

# Calculate the frequency distribution
product_frequency = Counter(product_data)

# Find the most popular product
most_popular_product = max(product_frequency, key=product_frequency.get)

# Print the frequency distribution and the most popular product
print("Product Frequency Distribution:")
for product, frequency in product_frequency.items():
    print(f"{product}: {frequency} times")

print(f"\nThe most popular product is: {most_popular_product}")
