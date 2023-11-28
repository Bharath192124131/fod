import matplotlib.pyplot as plt

# Replace these lists with your actual monthly sales data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales_values = [5000, 6000, 7500, 8000, 9000, 7000]

# Create a line plot
plt.figure(figsize=(8, 5))
plt.plot(months, sales_values, marker='o', linestyle='-', color='b')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.grid(True)
plt.show()




import matplotlib.pyplot as plt

# Replace these lists with your actual monthly sales data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
sales_values = [5000, 6000, 7500, 8000, 9000, 7000]

# Create a bar plot
plt.figure(figsize=(8, 5))
plt.bar(months, sales_values, color='green')
plt.title('Monthly Sales Data')
plt.xlabel('Month')
plt.ylabel('Sales ($)')
plt.grid(axis='y')
plt.show()
