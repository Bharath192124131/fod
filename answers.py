                                                set 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
age = [25, 30, 35, 40, 45]
salary = [50000, 60000, 75000, 90000, 110000]

# Creating a DataFrame
data = pd.DataFrame({'Age': age, 'Salary': salary})

# Correlation matrix
correlation_matrix = data.corr()

# Covariance matrix
covariance_matrix = data.cov()

# Plotting correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Plot')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have a diabetic dataset named 'diabetic_data'

# Bar graph for distribution
sns.countplot(x='variable', data=diabetic_data)
plt.title('Bar Graph for Variable Distribution')
plt.show()

# Line chart for variable
sns.lineplot(x='variable', y='value', data=diabetic_data)
plt.title('Line Chart for Variable')
plt.show()


import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Display DataFrame
print(df)

import numpy as np

# Assuming you have a NumPy array named 'fuel_efficiency'
fuel_efficiency = np.array([25, 30, 28, 35, 32])

# Calculate average fuel efficiency
average_fuel_efficiency = np.mean(fuel_efficiency)

# Determine percentage improvement between two car models
model1_efficiency = fuel_efficiency[0]
model2_efficiency = fuel_efficiency[1]

percentage_improvement = ((model2_efficiency - model1_efficiency) / model1_efficiency) * 100

# Display results
print(f'Average Fuel Efficiency: {average_fuel_efficiency}')
print(f'Percentage Improvement between Model 1 and Model 2: {percentage_improvement}%')



                                                   set 2

import pandas as pd
import numpy as np

# Sample data
data = {
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 75000, 90000, 110000]
}

# Creating DataFrame
df = pd.DataFrame(data)

# Pearson Correlation Coefficient
pearson_corr = df['age'].corr(df['salary'], method='pearson')

# Spearmans Rank Correlation Coefficient
spearman_corr = df['age'].corr(df['salary'], method='spearman')

# Displaying results
print(f'Pearson Correlation Coefficient: {pearson_corr}')
print(f'Spearmans Rank Correlation Coefficient: {spearman_corr}')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample housing dataset
housing_data = {
    'Square_Footage': [1500, 1800, 2000, 2200, 2500],
    'Price': [300000, 350000, 400000, 450000, 500000],
    'Number_of_Bedrooms': [3, 4, 3, 4, 5]
}

# Creating DataFrame
housing_df = pd.DataFrame(housing_data)

# Scatterplot
sns.scatterplot(x='Square_Footage', y='Price', data=housing_df)
plt.title('Scatterplot: Square Footage vs Price')
plt.show()

# Stacked Bar Chart
sns.barplot(x='Square_Footage', y='Price', hue='Number_of_Bedrooms', data=housing_df)
plt.title('Stacked Bar Chart: Square Footage, Price, Number of Bedrooms')
plt.show()


import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Get the first 3 rows
first_3_rows = df.head(3)

# Displaying the result
print('First 3 Rows of DataFrame:')
print(first_3_rows)


import pandas as pd

# Sample data
products_data = {
    'Product': ['A', 'B', 'A', 'C', 'B', 'C', 'A', 'B', 'C', 'A'],
    'Units_Sold': [30, 25, 35, 20, 28, 22, 33, 31, 18, 32]
}

# Creating DataFrame
products_df = pd.DataFrame(products_data)

# Calculating frequency distribution
product_frequency = products_df['Product'].value_counts()

# Finding the most popular product
most_popular_product = product_frequency.idxmax()

# Displaying results
print('Frequency Distribution of Products:')
print(product_frequency)
print(f'\nMost Popular Product: {most_popular_product}')

import pandas as pd
import numpy as np

# Sample dataset
data = np.random.normal(loc=50, scale=10, size=100)

# Mean estimation
mean_estimate = np.mean(data)

# Variance estimation
variance_estimate = np.var(data)

# Sampling technique (e.g., taking a random sample of 10 data points)
random_sample = np.random.choice(data, size=10, replace=False)

# Displaying results
print(f'Mean Estimation: {mean_estimate}')
print(f'Variance Estimation: {variance_estimate}')
print(f'Random Sample: {random_sample}')


                                              set 3 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sample data
age = [25, 30, 35, 40, 45]
salary = [50000, 60000, 75000, 90000, 110000]

# Creating a DataFrame
data = pd.DataFrame({'Age': age, 'Salary': salary})

# Correlation matrix
correlation_matrix = data.corr()

# Covariance matrix
covariance_matrix = data.cov()

# Plotting correlation heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Plot')
plt.show()


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sample car dataset
car_data = {
    'Car_Price': [30000, 35000, 40000, 45000, 50000],
    'Fuel_Efficiency': [25, 30, 28, 35, 32],
    'Horsepower': [200, 220, 250, 180, 210],
    'Weight': [3000, 3200, 3500, 2800, 3300]
}

# Creating DataFrame
car_df = pd.DataFrame(car_data)

# Multivariate Scatterplot
sns.scatterplot(x='Car_Price', y='Fuel_Efficiency', hue='Horsepower', size='Weight', data=car_df)
plt.title('Multivariate Scatterplot')
plt.show()

# Scatter Plot Matrix
sns.pairplot(car_df)
plt.suptitle('Scatter Plot Matrix', y=1.02)
plt.show()


import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Selecting rows with missing scores
missing_scores = df[df['score'].isna()]

# Displaying rows with missing scores
print('Rows with Missing Scores:')
print(missing_scores)


import pandas as pd

# Sample data
subjects_data = {
    'Subject': ['Math', 'English', 'Science', 'History', 'Math', 'English', 'Science', 'Math', 'History', 'Science'],
    'Students': [30, 25, 35, 20, 28, 22, 33, 31, 18, 32]
}

# Creating DataFrame
subjects_df = pd.DataFrame(subjects_data)

# Calculating frequency distribution
subject_frequency = subjects_df['Subject'].value_counts()

# Finding the most popular subject
most_popular_subject = subject_frequency.idxmax()

# Displaying results
print('Frequency Distribution of Subjects:')
print(subject_frequency)
print(f'\nMost Popular Subject: {most_popular_subject}')


                                                    set 4

import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Calculate the sum of examination attempts
total_attempts = df['attempts'].sum()

# Displaying the result
print(f'Total Examination Attempts: {total_attempts}')


import pandas as pd

# Sample data
weather_data = {
    'Weather_Condition': ['Sunny', 'Rainy', 'Cloudy', 'Sunny', 'Rainy', 'Sunny', 'Cloudy', 'Rainy', 'Sunny', 'Cloudy'],
    'Occurrences': [30, 25, 35, 20, 28, 22, 33, 31, 18, 32]
}

# Creating DataFrame
weather_df = pd.DataFrame(weather_data)

# Calculating frequency distribution
weather_frequency = weather_df['Weather_Condition'].value_counts()

# Finding the most common weather type
most_common_weather = weather_frequency.idxmax()

# Displaying results
print('Frequency Distribution of Weather Conditions:')
print(weather_frequency)
print(f'\nMost Common Weather Type: {most_common_weather}')


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.datasets import make_classification

# Generate a hypothetical dataset for demonstration
X, y = make_classification(n_samples=100, n_features=1, n_informative=1, n_redundant=0, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_model.fit(X, y)

# Visualize linear regression
sns.regplot(x=X[:, 0], y=y, fit_reg=True, scatter_kws={'s': 50})
plt.title('Linear Regression')
plt.show()

# Visualize logistic regression
sns.regplot(x=X[:, 0], y=y, logistic=True, scatter_kws={'s': 50})
plt.title('Logistic Regression')
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Assume you have a dataset named 'medical_dataset.csv'
# The dataset should have columns for symptoms and a column for the medical condition label (0 or 1)

# Load the dataset
dataset = pd.read_csv('medical_dataset.csv')

# Assuming 'X' contains features (symptoms) and 'y' contains labels (0 or 1)
X = dataset.drop('label', axis=1)
y = dataset['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input for new patient features
new_patient_features = [float(input(f'Enter value for {feature}: ')) for feature in X.columns]

# User input for k (number of neighbors)
k = int(input('Enter the value of k: '))

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training set
knn_classifier.fit(X_train, y_train)

# Predict the medical condition for the new patient
prediction = knn_classifier.predict([new_patient_features])

# Display the prediction
print(f'Predicted Medical Condition: {prediction[0]}')


                                                       set 5

import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Replace 'qualify' column values with True and False
df['qualify'] = df['qualify'].map({'yes': True, 'no': False})

# Displaying the updated DataFrame
print('Updated DataFrame:')
print(df)

import pandas as pd
import matplotlib.pyplot as plt

# Sample monthly sales data
sales_data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Sales': [15000, 18000, 20000, 22000, 25000, 28000, 30000, 32000, 35000, 38000, 40000, 42000]
}

# Creating DataFrame
sales_df = pd.DataFrame(sales_data)

# Line plot
plt.figure(figsize=(8, 5))
plt.plot(sales_df['Month'], sales_df['Sales'], marker='o', label='Monthly Sales')
plt.title('Monthly Sales Data - Line Plot')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()

# Bar chart
plt.figure(figsize=(8, 5))
plt.bar(sales_df['Month'], sales_df['Sales'], color='skyblue', label='Monthly Sales')
plt.title('Monthly Sales Data - Bar Chart')
plt.xlabel('Month')
plt.ylabel('Sales')
plt.legend()
plt.show()


import pandas as pd

# Sample data
accidents_data = {
    'Accident_ID': [1, 2, 3, 4, 5],
    'Cause': ['Speeding', 'Distracted Driving', 'Drunk Driving', 'Weather Conditions', 'Speeding']
}

# Creating DataFrame
accidents_df = pd.DataFrame(accidents_data)

# Calculating frequency distribution of accident causes
cause_frequency = accidents_df['Cause'].value_counts()

# Finding the most common cause of accidents
most_common_cause = cause_frequency.idxmax()

# Displaying results
print('Frequency Distribution of Accident Causes:')
print(cause_frequency)
print(f'\nMost Common Cause of Accidents: {most_common_cause}')
import pandas as pd

# Sample data
accidents_data = {
    'Accident_ID': [1, 2, 3, 4, 5],
    'Cause': ['Speeding', 'Distracted Driving', 'Drunk Driving', 'Weather Conditions', 'Speeding']
}

# Creating DataFrame
accidents_df = pd.DataFrame(accidents_data)

# Calculating frequency distribution of accident causes
cause_frequency = accidents_df['Cause'].value_counts()

# Finding the most common cause of accidents
most_common_cause = cause_frequency.idxmax()

# Displaying results
print('Frequency Distribution of Accident Causes:')
print(cause_frequency)
print(f'\nMost Common Cause of Accidents: {most_common_cause}')


import numpy as np

# Assuming you have a NumPy array named 'sales_data'
sales_data = np.array([50000, 60000, 75000, 90000])

# Calculate total sales for the year
total_sales = np.sum(sales_data)

# Calculate percentage increase from the first quarter to the fourth quarter
percentage_increase = ((sales_data[3] - sales_data[0]) / sales_data[0]) * 100

# Displaying results
print(f'Total Sales for the Year: {total_sales}')
print(f'Percentage Increase from Q1 to Q4: {percentage_increase}%')


                                                     set 6

import pandas as pd

# Sample data
exam_data = [{'name': 'Anastasia', 'score': 12.5},
             {'name': 'Dima', 'score': 9},
             {'name': 'Katherine', 'score': 16.5}]

# Creating DataFrame
df = pd.DataFrame(exam_data)

# Iterate over rows in the DataFrame
for index, row in df.iterrows():
    print(f"Name: {row['name']}, Score: {row['score']}")


import pandas as pd
import matplotlib.pyplot as plt

# Sample monthly temperature and rainfall data
weather_data = {
    'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
    'Temperature': [20, 22, 25, 28, 30, 32, 35, 33, 28, 25, 22, 20],
    'Rainfall': [50, 40, 30, 20, 15, 10, 5, 10, 15, 25, 30, 40]
}

# Creating DataFrame
weather_df = pd.DataFrame(weather_data)

# Scatter plot
plt.scatter(weather_df['Temperature'], weather_df['Rainfall'])
plt.title('Scatter Plot: Temperature vs Rainfall')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.show()

# Line plot
plt.plot(weather_df['Month'], weather_df['Rainfall'], marker='o', linestyle='-', color='blue', label='Rainfall')
plt.title('Line Plot: Monthly Rainfall Data')
plt.xlabel('Month')
plt.ylabel('Rainfall (mm)')
plt.legend()
plt.show()

import pandas as pd

# Sample data
diseases_data = {
    'Disease': ['Flu', 'Cancer', 'Diabetes', 'Flu', 'Cancer', 'Diabetes', 'Flu', 'Cancer', 'Asthma', 'Asthma'],
    'Patients': [30, 25, 35, 20, 28, 22, 33, 31, 18, 32]
}

# Creating DataFrame
diseases_df = pd.DataFrame(diseases_data)

# Calculating frequency distribution of diseases
disease_frequency = diseases_df['Disease'].value_counts()

# Finding the most common disease
most_common_disease = disease_frequency.idxmax()

# Displaying results
print('Frequency Distribution of Diseases:')
print(disease_frequency)
print(f'\nMost Common Disease: {most_common_disease}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume you have a dataset named 'your_dataset.csv'
# Load the dataset
dataset = pd.read_csv('your_dataset.csv')

# User input for features and target variable
features = input('Enter names of features (comma-separated): ').split(',')
target_variable = input('Enter the name of the target variable: ')

# Extract features and target variable from the dataset
X = dataset[features]
y = dataset[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input for model choice (here, using Decision Tree as an example)
model = DecisionTreeClassifier()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Displaying results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')


                                                        set 7

import pandas as pd

# Sample data for two series
series1 = pd.Series([1, 2, 3, 4], name='Series1')
series2 = pd.Series(['A', 'B', 'C', 'D'], name='Series2')

# Combine two series into a DataFrame
df = pd.DataFrame({'Series1': series1, 'Series2': series2})

# Displaying the DataFrame
print(df)

import scipy.stats as stats

# Assume you have conversion rate data for designs A and B
conversion_rate_A = [0.1, 0.12, 0.11, 0.09, 0.1, 0.11, 0.12]
conversion_rate_B = [0.14, 0.15, 0.13, 0.16, 0.14, 0.15, 0.13]

# Perform an independent t-test
t_stat, p_value = stats.ttest_ind(conversion_rate_A, conversion_rate_B)

# Determine statistical significance
alpha = 0.05
if p_value < alpha:
    print(f'There is a statistically significant difference in mean conversion rates (p-value: {p_value})')
else:
    print(f'No statistically significant difference in mean conversion rates (p-value: {p_value})')


import pandas as pd
import matplotlib.pyplot as plt

# Sample data for temperature and rainfall
weather_data = {
    'Temperature': [25, 28, 30, 22, 24, 26, 29, 31, 23, 27],
    'Rainfall': [50, 40, 30, 20, 15, 10, 5, 10, 15, 25]
}

# Creating DataFrame
weather_df = pd.DataFrame(weather_data)

# Calculate the correlation coefficient
correlation_coefficient = weather_df['Temperature'].corr(weather_df['Rainfall'])

# Create a scatter plot
plt.scatter(weather_df['Temperature'], weather_df['Rainfall'])
plt.title('Scatter Plot: Temperature vs Rainfall')
plt.xlabel('Temperature (°C)')
plt.ylabel('Rainfall (mm)')
plt.show()

# Displaying the correlation coefficient
print(f'Correlation Coefficient: {correlation_coefficient}')

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assume you have a dataset named 'medical_dataset.csv'
# Load the dataset
dataset = pd.read_csv('medical_dataset.csv')

# User input for features and target variable
features = dataset.columns[:-1]  # Assuming the last column is the target variable
target_variable = dataset.columns[-1]

# Extract features and target variable from the dataset
X = dataset[features]
y = dataset[target_variable]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User input for k (number of neighbors)
k = int(input('Enter the value of k for KNN: '))

# Initialize KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=k)

# Train the classifier on the training set
knn_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Good')
recall = recall_score(y_test, y_pred, pos_label='Good')
f1 = f1_score(y_test, y_pred, pos_label='Good')

# Displaying results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

                                                   set 8

import pandas as pd
import matplotlib.pyplot as plt

# Sample data for sales and advertising
sales_and_advertising_data = {
    'Sales': [100, 120, 150, 80, 90, 110, 130, 160, 140, 120],
    'Advertising_Spend': [50, 60, 75, 40, 45, 55, 65, 80, 70, 60]
}

# Creating DataFrame
sales_and_advertising_df = pd.DataFrame(sales_and_advertising_data)

# Calculate the correlation coefficient
correlation_coefficient = sales_and_advertising_df['Sales'].corr(sales_and_advertising_df['Advertising_Spend'])

# Create a scatter plot
plt.scatter(sales_and_advertising_df['Advertising_Spend'], sales_and_advertising_df['Sales'])
plt.title('Scatter Plot: Sales vs Advertising Spend')
plt.xlabel('Advertising Spend ($)')
plt.ylabel('Sales')
plt.show()

# Displaying the correlation coefficient
print(f'Correlation Coefficient: {correlation_coefficient}')

import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Select 'name' and 'score' columns in rows 1, 3, 5, 6
selected_data = df.loc[['b', 'd', 'f', 'g'], ['name', 'score']]

# Displaying the selected data
print(selected_data)


import pandas as pd
from scipy.stats import t

# Load customer reviews data
customer_reviews_df = pd.read_csv('customer_reviews.csv')

# Assuming 'rating' is the column containing customer ratings
ratings = customer_reviews_df['rating']

# Calculate confidence interval for the mean rating
confidence_level = 0.95
confidence_interval = t.interval(confidence_level, len(ratings)-1, loc=ratings.mean(), scale=ratings.sem())

# Displaying results
print(f'Average Rating: {ratings.mean()}')
print(f'Confidence Interval: {confidence_interval}')


import pandas as pd
import matplotlib.pyplot as plt

# Sample data for smoking and lung cancer
smoking_and_lung_cancer_data = {
    'Patients_Smoking': [100, 120, 150, 80, 90, 110, 130, 160, 140, 120],
    'Patients_Lung_Cancer': [10, 12, 15, 8, 9, 11, 13, 16, 14, 12]
}

# Creating DataFrame
smoking_and_lung_cancer_df = pd.DataFrame(smoking_and_lung_cancer_data)

# Calculate the correlation coefficient
correlation_coefficient = smoking_and_lung_cancer_df['Patients_Smoking'].corr(smoking_and_lung_cancer_df['Patients_Lung_Cancer'])

# Create a scatter plot
plt.scatter(smoking_and_lung_cancer_df['Patients_Smoking'], smoking_and_lung_cancer_df['Patients_Lung_Cancer'])
plt.title('Scatter Plot: Smoking vs Lung Cancer')
plt.xlabel('Patients Smoking')
plt.ylabel('Patients with Lung Cancer')
plt.show()

# Displaying the correlation coefficient
print(f'Correlation Coefficient: {correlation_coefficient}')


                                                            set 9

import pandas as pd

# Load stock data from CSV file
stock_data = pd.read_csv('stock_prices.csv')

# Assuming the column names are 'Date' and 'Closing Price'
closing_prices = stock_data['Closing Price']

# Calculate variability metrics
mean_price = closing_prices.mean()
std_dev = closing_prices.std()
price_range = closing_prices.max() - closing_prices.min()

# Determine stock price movements
if mean_price < closing_prices.iloc[-1]:
    trend = 'upward'
elif mean_price > closing_prices.iloc[-1]:
    trend = 'downward'
else:
    trend = 'stable'

# Displaying results
print(f'Mean Price: {mean_price}')
print(f'Standard Deviation: {std_dev}')
print(f'Price Range: {price_range}')
print(f'Trend: {trend}')

import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Select rows where the score is between 15 and 20 (inclusive)
selected_rows = df[(df['score'] >= 15) & (df['score'] <= 20)]

# Displaying the selected rows
print(selected_rows)

import pandas as pd
import matplotlib.pyplot as plt

# Sample data for smoking and lung cancer
smoking_and_lung_cancer_data = {
    'Patients_Smoking': [100, 120, 150, 80, 90, 110, 130, 160, 140, 120],
    'Patients_Lung_Cancer': [10, 12, 15, 8, 9, 11, 13, 16, 14, 12]
}

# Creating DataFrame
smoking_and_lung_cancer_df = pd.DataFrame(smoking_and_lung_cancer_data)

# Calculate the correlation coefficient
correlation_coefficient = smoking_and_lung_cancer_df['Patients_Smoking'].corr(smoking_and_lung_cancer_df['Patients_Lung_Cancer'])

# Create a scatter plot
plt.scatter(smoking_and_lung_cancer_df['Patients_Smoking'], smoking_and_lung_cancer_df['Patients_Lung_Cancer'])
plt.title('Scatter Plot: Smoking vs Lung Cancer')
plt.xlabel('Patients Smoking')
plt.ylabel('Patients with Lung Cancer')
plt.show()

# Displaying the correlation coefficient
print(f'Correlation Coefficient: {correlation_coefficient}')

import pandas as pd

# Load temperature data from CSV file
temperature_data = pd.read_csv('temperature_readings.csv')

# Assuming the column names are 'City' and 'Temperature'
mean_temperatures = temperature_data.groupby('City')['Temperature'].mean()
std_dev_temperatures = temperature_data.groupby('City')['Temperature'].std()
temperature_ranges = temperature_data.groupby('City')['Temperature'].max() - temperature_data.groupby('City')['Temperature'].min()

# Determine the city with the highest temperature range
city_highest_range = temperature_ranges.idxmax()

# Determine the city with the most consistent temperature (lowest standard deviation)
city_most_consistent = std_dev_temperatures.idxmin()

# Displaying results
print(f'Mean Temperatures by City:\n{mean_temperatures}')
print(f'Standard Deviations by City:\n{std_dev_temperatures}')
print(f'Temperature Ranges by City:\n{temperature_ranges}')
print(f'City with Highest Temperature Range: {city_highest_range}')
print(f'City with Most Consistent Temperature: {city_most_consistent}')


                                                        set 10


import pandas as pd
import numpy as np

# Sample data
exam_data = {
    'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
    'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
    'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
    'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']
}
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']

# Creating DataFrame
df = pd.DataFrame(exam_data, index=labels)

# Displaying the DataFrame
print(df)


import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
shoe_sales_data = pd.read_csv('shoe_sales.csv')

# Calculate frequency distribution of shoe sizes
shoe_size_distribution = shoe_sales_data.groupby('shoe_size')['quantity'].sum()

# Displaying frequency distribution table
print(shoe_size_distribution)

# Create a bar chart
shoe_size_distribution.plot(kind='bar', xlabel='Shoe Size', ylabel='Quantity', title='Shoe Size Frequency Distribution')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt

# Sample data for crime rate and poverty rate
crime_and_poverty_data = {
    'City': ['City A', 'City B', 'City C', 'City D', 'City E'],
    'Crime_Rate': [100, 120, 80, 150, 90],
    'Poverty_Rate': [10, 15, 8, 20, 12]
}

# Creating DataFrame
crime_and_poverty_df = pd.DataFrame(crime_and_poverty_data)

# Calculate the correlation coefficient
correlation_coefficient = crime_and_poverty_df['Crime_Rate'].corr(crime_and_poverty_df['Poverty_Rate'])

# Create a scatter plot
plt.scatter(crime_and_poverty_df['Poverty_Rate'], crime_and_poverty_df['Crime_Rate'])
plt.title('Scatter Plot: Crime Rate vs Poverty Rate')
plt.xlabel('Poverty Rate')
plt.ylabel('Crime Rate')
plt.show()

# Displaying the correlation coefficient
print(f'Correlation Coefficient: {correlation_coefficient}')


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming you have a dataset named 'car_data.csv' with relevant features and 'price' as the target variable

# Read data from CSV file
car_data = pd.read_csv('car_data.csv')

# Selecting features and target variable
features = car_data[['engine_size', 'horsepower', 'fuel_efficiency']]
target = car_data['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)

# Displaying results
print(f'Mean Squared Error: {mse}')

# Plotting predictions vs actual values
plt.scatter(y_test, predictions)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Prices vs Predicted Prices')
plt.show()






