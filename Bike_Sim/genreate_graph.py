import pandas as pd
import matplotlib.pyplot as plt

# Define the filename
filename = 'example.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(filename)

# Display the first few rows of the DataFrame
print("Data Preview:")
print(df.head())

# Plot the data
plt.figure(figsize=(14, 7))

# Plot steerCmd
plt.subplot(3, 1, 1)
plt.plot(df.index, df['steerCmd'], label='Steer Command', color='blue')
plt.xlabel('Index')
plt.ylabel('Steer Command')
plt.title('Steer Command Over Time')
plt.legend()
plt.grid(True)

# Plot brakeCmd
plt.subplot(3, 1, 2)
plt.plot(df.index, df['brakeCmd'], label='Brake Command', color='red')
plt.xlabel('Index')
plt.ylabel('Brake Command')
plt.title('Brake Command Over Time')
plt.legend()
plt.grid(True)

# Plot throttleCmd
plt.subplot(3, 1, 3)
plt.plot(df.index, df['throttleCmd'], label='Throttle Command', color='green')
plt.xlabel('Index')
plt.ylabel('Throttle Command')
plt.title('Throttle Command Over Time')
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
