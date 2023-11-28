import pandas as pd
import matplotlib.pyplot as plt

# Load shoe sales dataset (replace 'shoe_sales.csv' with your actual file path)
shoe_sales_df = pd.read_csv('shoe_sales.csv')

# Assuming the dataset has columns 'shoe_size' and 'quantity'
frequency_distribution = shoe_sales_df.groupby('shoe_size')['quantity'].sum()

# Display frequency distribution table
print("Frequency Distribution of Shoe Sizes:")
print(frequency_distribution)

# Create a bar chart
plt.bar(frequency_distribution.index, frequency_distribution)
plt.xlabel('Shoe Size')
plt.ylabel('Quantity Sold')
plt.title('Shoe Size Frequency Distribution')
plt.show()
