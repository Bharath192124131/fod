import numpy as np

# Assuming student_scores is a 4x4 NumPy array
# Replace this array with your actual data
student_scores = np.array([
    [85, 90, 78, 88],
    [92, 88, 75, 80],
    [78, 85, 92, 87],
    [90, 95, 89, 79]
])

# Calculate the average score for each subject (column-wise mean)
average_scores_per_subject = np.mean(student_scores, axis=0)

# Identify the subject with the highest average score
subject_with_highest_average = np.argmax(average_scores_per_subject)

# Print the results
print("Average Scores per Subject:", average_scores_per_subject)
print("Subject with the Highest Average Score:", subject_with_highest_average)
