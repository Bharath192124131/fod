correlation_per_course = student_data.groupby('Course')['Hours_Studied', 'Score'].corr().iloc[0::2, -1]
print("Correlation Coefficient for each course:")
print(correlation_per_course)
strongest_correlation = correlation_per_course.groupby('Course').idxmax().values.flatten()
weakest_correlation = correlation_per_course.groupby('Course').idxmin().values.flatten()

print("Courses with the Strongest Correlation:")
print(strongest_correlation)
print("\nCourses with the Weakest Correlation:")
print(weakest_correlation)
statistical_insights = student_data.groupby('Course').agg({'Score': ['mean', 'std', 'min', 'max'], 'Hours_Studied': ['mean', 'std', 'min', 'max']})
print("Statistical Insights:")
print(statistical_insights)
