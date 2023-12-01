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
