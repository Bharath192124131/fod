import pandas as pd
import matplotlib.pyplot as plt

# Create a hypothetical dataset
data = {
    'Player': ['Messi', 'Ronaldo', 'Neymar', 'Mbappe', 'Salah', 'Lewandowski', 'Kane', 'Benzema', 'Suarez', 'Aguero'],
    'Age': [34, 36, 29, 23, 29, 33, 28, 33, 35, 32],
    'Position': ['Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward', 'Forward'],
    'Goals_Scored': [30, 25, 20, 27, 22, 35, 31, 29, 24, 28],
    'Weekly_Salary': [500000, 450000, 400000, 300000, 350000, 380000, 410000, 420000, 470000, 440000]
}

# Create a DataFrame and save it to a CSV file
df = pd.DataFrame(data)
df.to_csv('soccer_players.csv', index=False)

# Read the data from the CSV file into a pandas DataFrame
df = pd.read_csv('soccer_players.csv')

# Task 1: Find the top 5 players with the highest number of goals scored
top_players_goals = df.nlargest(5, 'Goals_Scored')

# Task 2: Find the top 5 players with the highest salaries
top_players_salary = df.nlargest(5, 'Weekly_Salary')

# Task 3: Calculate the average age of players
average_age = df['Age'].mean()

# Task 4: Display the names of players above the average age
above_average_age_players = df[df['Age'] > average_age]['Player']

# Task 5: Visualize the distribution of players based on their positions using a bar chart
position_distribution = df['Position'].value_counts()

# Plot the bar chart
plt.figure(figsize=(10, 6))
position_distribution.plot(kind='bar', color='skyblue')
plt.title('Distribution of Players Based on Positions')
plt.xlabel('Position')
plt.ylabel('Number of Players')
plt.show()

# Display the results
print("\nTop 5 Players with the Highest Number of Goals:")
print(top_players_goals[['Player', 'Goals_Scored']])

print("\nTop 5 Players with the Highest Salaries:")
print(top_players_salary[['Player', 'Weekly_Salary']])

print(f"\nAverage Age of Players: {average_age:.2f}")

print("\nPlayers Above the Average Age:")
print(above_average_age_players)
