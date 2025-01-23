import os
import pandas as pd
import matplotlib.pyplot as plt

# Define the filename (modify as needed)
filename = r'C:\Users\CARLA-1\Desktop\project\carla\WindowsNoEditor\PythonAPI\project_bicycle_carla\another one_A.csv' #for Lab

# Check if the file exists
if os.path.exists(filename):
    # Read the CSV file into a Pandas DataFrame
    df_orig = pd.read_csv(filename)

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
    # Plot commands in a separate figure
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title(f"{title}: Commands")
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Steer'], label='Steer Command', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Steer Command')
    plt.title(f"{title}: Steer Command Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['Brake'], label='Brake Command', color='red')
    plt.xlabel('Index')
    plt.ylabel('Brake Command')
    plt.title(f"{title}: Brake Command Over Time")
    plt.legend()
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Throttle'], label='Throttle Command', color='green')
    plt.xlabel('Index')
    plt.ylabel('Throttle Command')
    plt.title(f"{title}: Throttle Command Over Time")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)


    # Plot distances in a separate figure
    fig = plt.figure(figsize=(14, len(distances) * 3))
    fig.canvas.manager.set_window_title(f"{title}: Distances")
    for i, distance in enumerate(distances, start=1):
        plt.subplot(len(distances), 1, i)
        plt.plot(df.index, df[distance], label=distance, color='black')
        plt.xlabel('Index')
        plt.ylabel(distance)
        plt.title(f"{title}: {distance} Over Time")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show(block=False)


# Plot additional data in their own windows
additional_data = [
    ('Speed_kmh', 'Speed (km/h)', 'Speed (km/h)', 'purple'),
    ('Heading', 'Heading', 'Heading', 'orange'),
    ('Height', 'Height', 'Height (m)', 'brown'),
    ('GNSS_Latitude', 'GNSS Latitude', 'Latitude', 'teal'),
    ('GNSS_Longitude', 'GNSS Longitude', 'Longitude', 'cyan'),
    ('Nearby_Vehicles_Count', 'Nearby Vehicles Count', 'Count', 'magenta'),
]

for col, title, ylabel, color in additional_data:
    fig = plt.figure(figsize=(14, 7))
    fig.canvas.manager.set_window_title(title)
    plt.plot(df_orig.index, df_orig[col], label=title, color=color)
    plt.xlabel('Index')
    plt.ylabel(ylabel)
    plt.title(f"{title} Over Time")
    plt.legend()
    plt.grid(True)
    plt.show(block=True)

