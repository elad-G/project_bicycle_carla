import pandas as pd
import matplotlib.pyplot as plt
import os

# Define the filename
filename = r'C:\Users\CARLA-1\Desktop\project\carla\WindowsNoEditor\PythonAPI\project_bicycle_carla\test_run_2150_B.csv' #for Lab
# filename = r'/Users/aliza/project_bicycle_carla/data_base/test_run_elad_B.csv' # for Elad's env
# Check if the file exists
if os.path.exists(filename):
    # Read the CSV file into a Pandas DataFrame
    df_orig = pd.read_csv(filename)

"""
    this script will:
    - if path ends with A:
        plot the graph
        calculate player steering variance through the drive

    - if path ends with B:  
        split the drive by distance from vehicle
        display 3 different graphs for each drive.

"""

df_distance1 = df_orig[df_orig['distance1']<100]
# plt.figure(figsize=(14, 7))
df_distance2 = df_orig[df_orig['distance2']<100]
df_distance3 = df_orig[df_orig['distance3']<100]
df_list = [df_orig,df_distance1,df_distance2,df_distance3]
# Loop through each DataFrame and create separate figures
for idx, df in enumerate(df_list, start=1):
    fig = plt.figure(figsize=(14, 18))  # Create a new figure for each DataFrame
    # Set a custom window title
    if idx ==1:
        fig.canvas.manager.set_window_title(f"Run Results")
        graph_num = 6
    elif idx ==2:
        graph_num = 4
        fig.canvas.manager.set_window_title(f"Bicycle")
    elif idx ==3:
        graph_num = 4
        fig.canvas.manager.set_window_title(f"MotorBike")
    elif idx ==4:
        graph_num = 4
        fig.canvas.manager.set_window_title(f"SmallCar")

    # Plot steerCmd
    plt.subplot(graph_num, 1, 1)
    plt.plot(df.index, df['steerCmd'], label='Steer Command', color='blue')
    plt.xlabel('Index')
    plt.ylabel('Steer Command')
    plt.title(' Steer Command Over Time')
    plt.legend()
    plt.grid(True)

    # Plot brakeCmd
    plt.subplot(graph_num, 1, 2)
    plt.plot(df.index, df['brakeCmd'], label='Brake Command', color='red')
    plt.xlabel('Index')
    plt.ylabel('Brake Command')
    plt.title(' Brake Command Over Time')
    plt.legend()
    plt.grid(True)

    # Plot throttleCmd
    plt.subplot(graph_num, 1, 3)
    plt.plot(df.index, df['throttleCmd'], label='Throttle Command', color='green')
    plt.xlabel('Index')
    plt.ylabel('Throttle Command')
    plt.title(' Throttle Command Over Time')
    plt.legend()
    plt.grid(True)

    if idx ==1: 
        # Plot distance1
        plt.subplot(graph_num, 1, 4)
        plt.plot(df.index, df['distance1'], label='Distance1', color='black')
        plt.xlabel('Index')
        plt.ylabel('Distance1')
        plt.title('Distance1 Over Time')
        plt.legend()
        plt.grid(True)

        # Plot distance2
        plt.subplot(graph_num, 1, 5)
        plt.plot(df.index, df['distance2'], label='Distance2', color='gray')
        plt.xlabel('Index')
        plt.ylabel('Distance2')
        plt.title('Distance2 Over Time')
        plt.legend()
        plt.grid(True)

        # Plot distance3
        plt.subplot(graph_num, 1, 6)
        plt.plot(df.index, df['distance3'], label='Distance3', color='yellow')
        plt.xlabel('Index')
        plt.ylabel('Distance3')
        plt.title('Distance3 Over Time')
        plt.legend()
        plt.grid(True)
    else:
        if idx ==2:
            distance = 'distance1'
        elif idx ==3:
            distance = 'distance2'
        elif idx ==4:
            distance = 'distance3'
        # Plot 
        plt.subplot(graph_num, 1, 4)
        plt.plot(df.index, df[distance], label=distance, color='black')
        plt.xlabel('Index')
        plt.ylabel(distance)
        plt.title('Distance1 Over Time')
        plt.legend()
        plt.grid(True)


    # Adjust layout for each figure
    plt.tight_layout()

# Show all figures at the same time
plt.show()
