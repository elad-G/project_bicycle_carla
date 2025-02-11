import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename
filename = r'C:\Users\CARLA-1\Desktop\project\carla\WindowsNoEditor\PythonAPI\project_bicycle_carla\new_data_base\deeeeeeelte_B.csv' #for Lab
# filename = r'/Users/aliza/project_bicycle_carla/new_data_base/roee_run_B.csv' # for Elad's env
# Check if the file exists
# Check if the file exists
if os.path.exists(filename):
    # Read the CSV file into a Pandas DataFrame
    df_orig = pd.read_csv(filename)
print(df_orig)
# Filter DataFrames based on distance thresholds
df_distance_bicycle = df_orig[df_orig['Distance_Bicycle'] < 100]
df_distance_motorbike = df_orig[df_orig['Distance_Motorbike'] < 100]
df_distance_smallcar = df_orig[df_orig['Distance_SmallCar'] < 100]

# List of DataFrames for processing
df_list = [
    (df_orig, "Run Results", ["Distance_Bicycle", "Distance_Motorbike", "Distance_SmallCar"]),
    (df_distance_bicycle, "Bicycle", ["Distance_Bicycle"]),
    (df_distance_motorbike, "MotorBike", ["Distance_Motorbike"]),
    (df_distance_smallcar, "SmallCar", ["Distance_SmallCar"]),
]

# Loop through each DataFrame and create separate figures
for idx, (df, title, distances) in enumerate(df_list, start=1):
    fig = plt.figure(figsize=(14, 18))  # Create a new figure for each DataFrame
    fig.canvas.manager.set_window_title(title)

    # Plot steerCmd
    plt.subplot(len(distances) + 3, 1, 1)
    plt.plot(df.index, df['Steer'], label='Steer Command', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Steer Command')
    plt.title('Steer Command Over Time')
    plt.legend()
    plt.grid(True)

    # Plot brakeCmd
    plt.subplot(len(distances) + 3, 1, 2)
    plt.plot(df.index, df['Brake'], label='Brake Command', color='red')
    plt.xlabel('Index')
    plt.ylabel('Brake Command')
    plt.title('Brake Command Over Time')
    plt.legend()
    plt.grid(True)

    # Plot throttleCmd
    plt.subplot(len(distances) + 3, 1, 3)
    plt.plot(df.index, df['Throttle'], label='Throttle Command', color='green')
    plt.xlabel('Index')
    plt.ylabel('Throttle Command')
    plt.title('Throttle Command Over Time')
    plt.legend()
    plt.grid(True)

    # Plot distances
    for i, distance in enumerate(distances, start=4):
        plt.subplot(len(distances) + 3, 1, i)
        plt.plot(df.index, df[distance], label=distance, color='black')
        plt.xlabel('Index')
        plt.ylabel(distance)
        plt.title(f'{distance} Over Time')
        plt.legend()
        plt.grid(True)

    # Adjust layout for each figure
    plt.tight_layout()

# Show all figures at the same time
plt.show()
