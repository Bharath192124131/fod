import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assume you have a car dataset with columns: 'mpg', 'horsepower', 'weight', 'acceleration', 'origin'
# Load your dataset (replace 'car_data.csv' with your actual file path)
df = pd.read_csv('car_data.csv')

# Multivariate Scatterplot
sns.set(style="ticks")
sns.pairplot(df, vars=['mpg', 'horsepower', 'weight', 'acceleration'], hue='origin', markers=["o", "s", "D"])

# Scatter Plot Matrix
sns.set(style="whitegrid")
sns.pairplot(df, diag_kind='kde', hue='origin')

plt.show()
