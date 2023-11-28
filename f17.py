#pip install pandas scikit-learn matplotlib

import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load your transaction data (replace 'your_data.csv' with the actual file path)
# The dataset should have customer IDs, total amount spent, and frequency of visits
data = pd.read_csv('your_data.csv')

# Select relevant features for clustering (total amount spent and frequency of visits)
X = data[['total_amount_spent', 'frequency_of_visits']]

# Choose the number of clusters (you can adjust this based on your business needs)
num_clusters = 3

# Build the K-Means model
kmeans_model = KMeans(n_clusters=num_clusters, random_state=42)
kmeans_model.fit(X)

# Add a new column to the original data indicating the cluster assignment for each customer
data['cluster'] = kmeans_model.labels_

# Visualize the clusters
plt.figure(figsize=(10, 6))
plt.scatter(data['total_amount_spent'], data['frequency_of_visits'], c=data['cluster'], cmap='viridis', alpha=0.7)
plt.title('Customer Segmentation using K-Means')
plt.xlabel('Total Amount Spent')
plt.ylabel('Frequency of Visits')
plt.show()
