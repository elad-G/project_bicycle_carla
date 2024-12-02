import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename
filename = r'C:\Users\CARLA-1\Desktop\project\carla\WindowsNoEditor\PythonAPI\project_bicycle_carla\log.csv'

# Check if the file exists
if os.path.exists(filename):
    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename)


# Read the CSV file into a DataFrame
# df = pd.read_csv(filename)

# # Merge each consecutive pair of rows
# merged_rows = []

# # Iterate through the rows in steps of 2
# for i in range(0, len(df), 2):
#     # Combine the two consecutive rowspip3 install 
#     row1 = df.iloc[i]
#     row2 = df.iloc[i + 1] if i + 1 < len(df) else None
#     if row2 is not None:
#         merged_row = row1.copy()
#         # Average or combine the values as needed (this example just takes row1 for the values)
#         merged_row['distance'] = (row1['distance'] + row2['distance']) / 2
#         merged_rows.append(merged_row)

# # Create a new DataFrame for the merged rows
# merged_df = pd.DataFrame(merged_rows)

# # Show the merged DataFrame
# print(merged_df)


# df = merged_df

# Display the first few rows of the DataFrame
print("Data Preview:")
print(os.getcwd())

# Plot the data
plt.figure(figsize=(14, 7))

# Plot steerCmd
plt.subplot(5, 1, 1)
plt.plot(df.index, df['steerCmd'], label='Steer Command', color='blue')
plt.xlabel('Index')
plt.ylabel('Steer Command')
plt.title('Steer Command Over Time')
plt.legend()
plt.grid(True)

# Plot brakeCmd
plt.subplot(5, 1, 2)
plt.plot(df.index, df['brakeCmd'], label='Brake Command', color='red')
plt.xlabel('Index')
plt.ylabel('Brake Command')
plt.title('Brake Command Over Time')
plt.legend()
plt.grid(True)

# Plot throttleCmd
plt.subplot(5, 1, 3)
plt.plot(df.index, df['throttleCmd'], label='Throttle Command', color='green')
plt.xlabel('Index')
plt.ylabel('Throttle Command')
plt.title('Throttle Command Over Time')
plt.legend()
plt.grid(True)

# Plot distance
plt.subplot(5, 1, 4)
plt.plot(df.index, df['distance1'], label='distance1', color='black')
plt.xlabel('Index')
plt.ylabel('distance')
plt.title('distance Over Time')
plt.legend()
plt.grid(True)

# Plot distance
plt.subplot(5, 1, 5)
plt.plot(df.index, df['distance2'], label='distance2', color='gray')
plt.xlabel('Index')
plt.ylabel('distance')
plt.title('distance Over Time')
plt.legend()
plt.grid(True)


# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()
