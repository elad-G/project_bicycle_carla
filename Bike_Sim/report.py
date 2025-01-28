import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def add_accumulative_time_delta_column(df, time_column, new_column_name='Accumulated_Time_Delta_ms'):
    """
    Adds a column of accumulated time deltas in milliseconds to the DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the time column.
        time_column (str): The name of the column with time data (format: HH:MM:SS:fff).
        new_column_name (str): The name of the new column to store accumulated time deltas in milliseconds.
    
    Returns:
        pd.DataFrame: The DataFrame with the new column added.
    """
    # Convert the specified time column to datetime format
    df[time_column] = pd.to_datetime(df[time_column], format='%H:%M:%S:%f')
    
    # Calculate the time deltas in milliseconds
    df['Time_Delta_ms'] = df[time_column].diff().dt.total_seconds() * 1000
    
    # Fill NaN (from the diff operation for the first row) with 0
    df['Time_Delta_ms'] = df['Time_Delta_ms'].fillna(0)
    
    # Calculate the accumulated deltas
    df[new_column_name] = df['Time_Delta_ms'].cumsum()
    
    # Drop the intermediate column if not needed
    df = df.drop(columns=['Time_Delta_ms'])

    return df


# # Function to calculate lateral deviation
# def calculate_deviation(df, entity_x, entity_y):
#     """Calculate the lateral deviation of the vehicle from a fixed entity location."""
#     df['Deviation'] = np.sqrt((df['Location_X'] - entity_x)**2 + (df['Location_Y'] - entity_y)**2)
#     return df

# # Function to create unified graph display
# def create_combined_graphs(graphs, title):
#     """Display multiple graphs in subplots within a single figure."""
#     fig, axes = plt.subplots(len(graphs), 1, figsize=(12, len(graphs) * 4))
#     fig.suptitle(title, fontsize=16)

#     for ax, graph in zip(axes, graphs):
#         label = graph['label']
#         data = graph['data']
#         x_label = graph['x_label']
#         y_label = graph['y_label']

#         ax.plot(data['x'], data['y'], label=label)
#         ax.set_title(label)
#         ax.set_xlabel(x_label)
#         ax.set_ylabel(y_label)
#         ax.legend()
#         ax.grid()

#     plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the title
#     plt.show()

# # Generalized analysis function
# def analyze_entity(df, entity_name, entity_x, entity_y, time_column):
#     """Generalized function for analyzing entities and creating graphs."""
#     df = calculate_deviation(df, entity_x, entity_y)
#     return {
#         'label': f'Lateral Deviation from {entity_name}',
#         'data': {'x': df[time_column], 'y': df['Deviation']},
#         'x_label': time_column.capitalize(),
#         'y_label': 'Deviation (meters)'
#     }

# # Steering analysis
# def analyze_steering(df, time_column):
#     """Analyze steering behavior."""
#     return {
#         'label': 'Steering Behavior Over Time',
#         'data': {'x': df[time_column], 'y': df['Steer']},
#         'x_label': time_column.capitalize(),
#         'y_label': 'Steering Angle'
#     }

# # Proximity analysis
# def analyze_proximity(df, time_column):
#     """Analyze proximity to other entities."""
#     return [
#         {
#             'label': 'Distance to Bicycle',
#             'data': {'x': df[time_column], 'y': df['Distance_Bicycle']},
#             'x_label': time_column.capitalize(),
#             'y_label': 'Distance (meters)'
#         },
#         {
#             'label': 'Distance to Motorbike',
#             'data': {'x': df[time_column], 'y': df['Distance_Motorbike']},
#             'x_label': time_column.capitalize(),
#             'y_label': 'Distance (meters)'
#         },
#         {
#             'label': 'Distance to SmallCar',
#             'data': {'x': df[time_column], 'y': df['Distance_SmallCar']},
#             'x_label': time_column.capitalize(),
#             'y_label': 'Distance (meters)'
#         }
#     ]

# # Comparison between control and distraction runs
# def compare_runs(control_df, distraction_df, time_column):
#     """Compare control and distraction runs."""
#     comparisons = []
#     metrics = ['Steer', 'Distance_Bicycle', 'Distance_Motorbike', 'Distance_SmallCar']
#     for metric in metrics:
#         comparisons.append({
#             'label': f'Control Run: {metric}',
#             'data': {
#                 'x': control_df[time_column],
#                 'y': control_df[metric]
#             },
#             'x_label': time_column.capitalize(),
#             'y_label': metric
#         })
#         comparisons.append({
#             'label': f'Distraction Run: {metric}',
#             'data': {
#                 'x': distraction_df[time_column],
#                 'y': distraction_df[metric]
#             },
#             'x_label': time_column.capitalize(),
#             'y_label': metric
#         })
#     return comparisons

# Terminal report
def generate_terminal_report(df, entity_name, entity_x, entity_y):
    """Generate a report summarizing the analysis results."""
    df = calculate_deviation(df, entity_x, entity_y)
    steering_std = df['Steer'].std()
    avg_distance = df['Deviation'].mean()
    min_distance = df['Deviation'].min()

    print("\n=== Vehicle Behavior Analysis Report ===")
    print(f"Entity Analyzed: {entity_name}")
    print(f"Average Distance to {entity_name}: {avg_distance:.2f} meters")
    print(f"Minimum Distance to {entity_name}: {min_distance:.2f} meters")
    print(f"Steering Variability (Standard Deviation): {steering_std:.2f}")
    print("=======================================\n")

def plot_from_dataframes(df1, df2, x_col, y_col,title="", labels=('Dataset 1', 'Dataset 2')):
    """
    Plot data from two DataFrames on a single graph.

    Parameters:
    df1 (DataFrame): The first DataFrame.
    df2 (DataFrame): The second DataFrame.
    x_col (str): The column name for the x-axis.
    y_col (str): The column name for the y-axis.
    labels (tuple): Labels for the datasets (default: ('Dataset 1', 'Dataset 2')).

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))

    # Plot data from the first DataFrame
    plt.plot(df1[x_col], df1[y_col], label=labels[0], marker='o')

    # Plot data from the second DataFrame
    plt.plot(df2[x_col], df2[y_col], label=labels[1], marker='x')

    # Adding labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)

    # Adding grid and legend
    plt.grid(True)
    plt.legend()
# 
    plt.tight_layout()
    # plt.show(block=False)

def plot_2_graphs_from_dataframe(df1, x_col, y_col_1,y_col_2,title="",y_col_1_label='label_1', y_col_2_label='label_2' ):
    plt.figure(figsize=(10, 6))
    # Plot data from the first DataFrame
    plt.plot(df1[x_col], df1[y_col_1], label=y_col_1_label, marker='o')

    # Plot data from the second DataFrame
    plt.plot(df1[x_col], df1[y_col_2], label=y_col_2_label, marker='x')

    # Adding labels and title
    plt.xlabel(x_col)
    plt.ylabel(y_col_1_label+" and "+y_col_2_label)
    plt.title(title)

    # Adding grid and legend
    plt.grid(True)
    plt.legend()
# 
    plt.tight_layout()
    # plt.show(block=False)

def plot_from_dataframes_zoomed_with_min_line(df1, df2, x_col, y_col, filter_col, filter_range,title='title', labels=('Dataset 1', 'Dataset 2')):
    # Filter DataFrames based on the range in filter_col
    filtered_df1 = df1[(df1[filter_col] >= filter_range[0]) & (df1[filter_col] <= filter_range[1])]
    filtered_df2 = df2[(df2[filter_col] >= filter_range[0]) & (df2[filter_col] <= filter_range[1])]

    # Determine the minimum x value from the filtered DataFrames
    idx_min_x1 = filtered_df1[filter_col].idxmin()  if not filtered_df1.empty else None
    idx_min_x2 = filtered_df2[filter_col].idxmin()  if not filtered_df2.empty else None
    min_x1 = filtered_df1.loc[idx_min_x1, x_col]
    min_x2 = filtered_df2.loc[idx_min_x2, x_col]

    # Plot data from the filtered DataFrames
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_df1[x_col], filtered_df1[y_col], label=f'Filtered {labels[0]}', marker='o')
    plt.plot(filtered_df2[x_col], filtered_df2[y_col], label=f'Filtered {labels[1]}', marker='x')

    # Add vertical lines for the minimum x values
    if min_x1 is not None:
        plt.axvline(x=min_x1, color='red', linestyle='--', label=f'Min {labels[0]}: {min_x1}')
    if min_x2 is not None:
        plt.axvline(x=min_x2, color='blue', linestyle='--', label=f'Min {labels[1]}: {min_x2}')

    # Adding labels, title, and legend
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(title)
    plt.grid(True)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    # plt.show()

# Main function
def main():
    # Load the CSV files
    control_file = r'/Users/aliza/project_bicycle_carla/new_data_base/roee_run_A.csv'
    distraction_file = r'/Users/aliza/project_bicycle_carla/new_data_base/roee_run_B.csv'

    try:
        control_df = pd.read_csv(control_file)
        distraction_df = pd.read_csv(distraction_file)
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Convert time units
    add_accumulative_time_delta_column(control_df, 'time', new_column_name='Time_Delta_ms')
    add_accumulative_time_delta_column(distraction_df, 'time', new_column_name='Time_Delta_ms')


    # # plot steer command
    # plot_from_dataframes(control_df, distraction_df, 'Time_Delta_ms', 'Steer',"steer command over time", labels=('A', 'B'))
    # plot_from_dataframes(control_df, distraction_df, 'tick', 'Steer',"steer command over ticks", labels=('A', 'B'))

    # # plot ditance from bicycle
    # plot_from_dataframes(control_df, distraction_df, 'Time_Delta_ms', 'Distance_Bicycle',"Distance_Bicycle command over time", labels=('A', 'B'))
    # plot_from_dataframes(control_df, distraction_df, 'tick', 'Distance_Bicycle',"Distance_Bicycle command over ticks", labels=('A', 'B'))

    # # plot ditance from motorbike
    # plot_from_dataframes(control_df, distraction_df, 'Time_Delta_ms', 'Distance_Motorbike',"Distance_Motorbike command over time", labels=('A', 'B'))
    # plot_from_dataframes(control_df, distraction_df, 'tick', 'Distance_Motorbike',"Distance_Motorbike command over ticks", labels=('A', 'B'))
    # # plot ditance from small car
    # plot_from_dataframes(control_df, distraction_df, 'Time_Delta_ms', 'Distance_SmallCar',"Distance_SmallCar command over time", labels=('A', 'B'))
    # plot_from_dataframes(control_df, distraction_df, 'tick', 'Distance_SmallCar', "Distance_SmallCar command over ticks",labels=('A', 'B'))

    # # plot steer command zoomed
    # plot_from_dataframes_zoomed_with_min_line(control_df, distraction_df, 'Time_Delta_ms',  'Steer', 'Distance_Bicycle', (0,50),'Zoomed Comparison steer over Time', labels=('A', 'B'))
    # plot_from_dataframes_zoomed_with_min_line(control_df, distraction_df, 'tick',  'Steer', 'Distance_Bicycle', (0,50),'Zoomed Comparison steer over Ticks', labels=('A', 'B'))
    
    # # plot yaw command zoomed
    # plot_from_dataframes_zoomed_with_min_line(control_df, distraction_df, 'Time_Delta_ms',  'Rotation_Yaw', 'Distance_Bicycle', (0,50),'Zoomed Comparison Rotation_Yaw over Time', labels=('A', 'B'))
    # plot_from_dataframes_zoomed_with_min_line(control_df, distraction_df, 'tick',  'Rotation_Yaw', 'Distance_Bicycle', (0,50),'Zoomed Comparison Rotation_Yaw over Ticks', labels=('A', 'B'))
    
    plot_2_graphs_from_dataframe(control_df, 'Time_Delta_ms', 'Steer','Rotation_Yaw',title="steer and yaw control",y_col_1_label='Steer', y_col_2_label='Rotation_Yaw' )


    
    plt.show()

if __name__ == "__main__":
    main()
