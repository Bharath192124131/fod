import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

# Load your dataset
df = pd.read_csv('ecommerce_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Extract features and target variable
X = df[['age', 'income', 'browsing_duration', 'device_type']]
y = df['purchase']

# One-hot encode the 'device_type' variable
encoder = OneHotEncoder(sparse=False, drop='first')
X_encoded = pd.DataFrame(encoder.fit_transform(X[['device_type']]), columns=encoder.get_feature_names(['device_type']))
X = pd.concat([X, X_encoded], axis=1).drop(['device_type'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Example prediction for a new customer
new_customer = pd.DataFrame({'age': [25], 'income': [50000], 'browsing_duration': [30], 'device_type': ['mobile']})

# One-hot encode 'device_type' for the new customer
new_customer_encoded = pd.DataFrame(encoder.transform(new_customer[['device_type']]), columns=encoder.get_feature_names(['device_type']))
new_customer = pd.concat([new_customer, new_customer_encoded], axis=1).drop(['device_type'], axis=1)

# Make predictions
prediction = dt_classifier.predict(new_customer)
print(f"Prediction for the new customer: {prediction[0]}")
