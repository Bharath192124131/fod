import numpy as np

# Replace these arrays with your actual time intervals and vertical positions
time_intervals = np.array([0, 1, 2, 3, 4])  # Example time intervals
vertical_positions = np.array([0, 10, 20, 15, 5])  # Example vertical positions

# Calculate the change in vertical position
delta_vertical_positions = np.diff(vertical_positions)

# Calculate the change in time
delta_time_intervals = np.diff(time_intervals)

# Calculate the average velocity
average_velocity = np.mean(delta_vertical_positions / delta_time_intervals)

# Print the result
print("Average Velocity of the Projectile:", average_velocity)
