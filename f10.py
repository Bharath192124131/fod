import matplotlib.pyplot as plt

# Replace these lists with your actual monthly temperature data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
temperature_values = [20, 22, 25, 28, 30, 32, 34, 32, 30, 27, 24, 22]

# Create a line plot for temperature
plt.figure(figsize=(8, 5))
plt.plot(months, temperature_values, marker='o', linestyle='-', color='red')
plt.title('Monthly Temperature Data')
plt.xlabel('Month')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

# Replace these lists with your actual monthly rainfall data
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
rainfall_values = [50, 40, 60, 30, 70, 80, 90, 60, 40, 30, 20, 50]

# Create a scatter plot for rainfall
plt.figure(figsize=(8, 5))
plt.scatter(months, rainfall_values, color='blue', marker='o')
plt.title('Monthly Rainfall Data')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.grid(True)
plt.show()
