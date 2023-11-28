import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your transaction data (replace 'transaction_data.csv' with the actual file path)
# The dataset should have customer IDs, total amount spent, and the number of items purchased
data = pd.read_csv('transaction_data.csv')

# Select relevant features for clustering (total amount spent and number of items purchased)
X = data[['total_amount_spent', 'num_items_purchased']]

# Standardize the features to have zero mean and unit variance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Choose the number of clusters (you can adjust this based on your business needs)
num_clusters = 3

# Build the K-Means model
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_model.fit(X_scaled)

# Add a new column to the original data indicating the cluster assignment for each customer
data['cluster'] = kmeans_model.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['total_amount_spent'], data['num_items_purchased'], c=data['cluster'], cmap='viridis', alpha=0.7)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Total Amount Spent')
plt.ylabel('Number of Items Purchased')
plt.show()
