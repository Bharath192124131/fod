import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('fruit_dataset.csv')

# Define features and target variable
X = df[['weight', 'color', 'texture']]
y = df['type']

# Preprocessing pipeline for numerical and categorical features
numeric_features = ['weight']
categorical_features = ['color', 'texture']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Choose the optimal value of 'k' using cross-validation
for k in range(1, 11):
    model = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', KNeighborsClassifier(n_neighbors=k))])

    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f'k={k}, Mean Accuracy: {scores.mean()}')

# Train the final model with the chosen value of 'k'
final_model = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', KNeighborsClassifier(n_neighbors=3))])

final_model.fit(X, y)

# Example prediction
new_fruit = {'weight': 150, 'color': 'red', 'texture': 'smooth'}
predicted_type = final_model.predict(pd.DataFrame([new_fruit]))
print(f'Predicted Fruit Type: {predicted_type[0]}')
