import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Load the dataset (replace 'credit_data.csv' with the actual file path)
df = pd.read_csv('credit_data.csv')

# Display the first few rows of the dataset
print(df.head())

# Extract features and target variable
X = df[['income', 'credit_score', 'debt_to_income_ratio', 'employment_duration']]
y = df['risk']

# Convert categorical features to numerical if needed

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a CART classifier
cart_classifier = DecisionTreeClassifier(random_state=42)

# Train the model
cart_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = cart_classifier.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Example for a new loan applicant
new_applicant = pd.DataFrame({
    'income': [50000],
    'credit_score': [700],
    'debt_to_income_ratio': [0.3],
    'employment_duration': [2]
})

# Make predictions for the new applicant
new_applicant_risk = cart_classifier.predict(new_applicant)
print(f"Predicted Credit Risk for the New Applicant: {new_applicant_risk[0]}")
