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


# Function to calculate lateral deviation
def calculate_deviation(df, entity_x, entity_y):
    """Calculate the lateral deviation of the vehicle from a fixed entity location."""
    df['Deviation'] = np.sqrt((df['Location_X'] - entity_x)**2 + (df['Location_Y'] - entity_y)**2)
    return df


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



def plott(df,x_1_col,X_2_col,y_1_col,title = "no title", unit_top = '',unit_bot= ' ', unit_y = ''):
    fig2, ax12 = plt.subplots()
    ax12.plot(df[x_1_col], df[y_1_col], marker='o', linestyle='-', color='orange', label="Main Y Datsa")
    ax12.set_xlabel(x_1_col + unit_bot)
    ax12.set_ylabel(y_1_col + unit_y)
    ax12.tick_params(axis='y', labelcolor='orange')
    ax22 = ax12.twiny()    # Create secondary X-axis
    ax22.set_xlim(ax12.get_xlim())  # Match the X-axis limits
    x_ticks = ax12.get_xticks()    # Get the tick positions of the bottom X-axis
    x_ticks_interpolation = np.interp(x_ticks, df[x_1_col], df[X_2_col])    # Interpolate corresponding Time_Delta_ms values
    # middle_x_value = (ax22.get_xlim()[0] + ax22.get_xlim()[1]) / 2
    # ax12.axvline(x=middle_x_value, color='red', linestyle='--', linewidth=1.5, label="Middle Line")
    ax22.set_xticks(x_ticks)    # Apply the interpolated values as labels
    ax22.set_xticklabels(x_ticks_interpolation.astype(int), rotation=30, ha='center')
    ax22.set_xlabel(X_2_col+ unit_top)
    ax12.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(title)

def plot_2_plots(df,df2,x_1_col,X_2_col,y_1_col ,title = "no title"):
    fig2, ax12 = plt.subplots()
    fig2.figure(figsize=(10, 6))
    ax12.plot(df[x_1_col], df[y_1_col], marker='o', linestyle='-', color='orange', label="Main Y Data")
    ax12.plot(df2[x_1_col], df[y_1_col], marker='o', linestyle='-', color='orange', label="Main Y Data")
    ax12.set_xlabel(x_1_col)
    ax12.set_ylabel('y_1_col')
    ax12.tick_params(axis='y', labelcolor='orange')
    ax22 = ax12.twiny()    # Create secondary X-axis
    ax22.set_xlim(ax12.get_xlim())  # Match the X-axis limits
    x_ticks = ax12.get_xticks()    # Get the tick positions of the bottom X-axis
    x_ticks_interpolation = np.interp(x_ticks, df[x_1_col], df[X_2_col])    # Interpolate corresponding Time_Delta_ms values
    # middle_x_value = (ax22.get_xlim()[0] + ax22.get_xlim()[1]) / 2
    # ax12.axvline(x=middle_x_value, color='red', linestyle='--', linewidth=1.5, label="Middle Line")
    ax22.set_xticks(x_ticks)    # Apply the interpolated values as labels
    ax22.set_xticklabels(x_ticks_interpolation.astype(int), rotation=30, ha='center')
    ax22.set_xlabel(X_2_col)
    plt.legend()
    ax12.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    plt.title(title)


def plot_from_dataframes(df1, df2, x_col, y_col,title="", labels=('Dataset 1', 'Dataset 2')):
    plt.figure(figsize=(10, 6))
    plt.plot(df1[x_col], df1[y_col], label=labels[0], marker='o')
    plt.plot(df2[x_col], df2[y_col], label=labels[1], marker='x')
    plt.xlabel(x_col + "[m]")
    plt.ylabel(y_col + "[deg]")
    plt.title(title)
    vertical_lines = [-271.1+433.9,-67.2+433.9,147.2+433.9]
    if vertical_lines:
        for x_val in vertical_lines:
            plt.axvline(x=x_val, color='black', linestyle='--', linewidth=1.5)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

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

    # create accumulative x axis.
    df = control_df
    df['neg_steer'] = -df['Steer']    
    df['Location_X_tmp'] = df['Location_X'].diff()
    df['Location_X_tmp'] = df['Location_X_tmp'].fillna(0)
    df['x_axis'] = df['Location_X_tmp'].cumsum()
    df = df.drop(columns=['Location_X_tmp'])
    control_df = df
    df = distraction_df
    df['neg_steer'] = -df['Steer']    
    df['Location_X_tmp'] = df['Location_X'].diff()
    df['Location_X_tmp'] = df['Location_X_tmp'].fillna(0)
    df['x_axis'] = df['Location_X_tmp'].cumsum()
    df = df.drop(columns=['Location_X_tmp'])
    distraction_df = df

    spawn_point_bicycle_x = -271.1
    spawn_point_motorbike_x = -67.2
    spawn_point_smallcar_x = 147.2
    spawn_point_bicycle_y = 37.1
    spawn_point_motorbike_y = 37.3
    spawn_point_smallcar_y = 38.6

    distraction_df['y_delta_bicycle'] = spawn_point_bicycle_y - distraction_df['Location_Y']
    distraction_df['y_delta_motorbike'] = spawn_point_motorbike_y - distraction_df['Location_Y']
    distraction_df['y_delta_smallcar'] = spawn_point_smallcar_y - distraction_df['Location_Y']
    control_df['y_delta_bicycle'] = spawn_point_bicycle_y - control_df['Location_Y']
    control_df['y_delta_motorbike'] = spawn_point_motorbike_y - control_df['Location_Y']
    control_df['y_delta_smallcar'] = spawn_point_smallcar_y - control_df['Location_Y']


    df_smallcar_distraction = distraction_df[(distraction_df['Distance_SmallCar'] >= -100) & (distraction_df['Distance_SmallCar'] <= 100)]
    df_bicycle_distraction = distraction_df[(distraction_df['Distance_Bicycle'] >= -100) & (distraction_df['Distance_Bicycle'] <= 100)]
    df_Motorbike_distraction= distraction_df[(distraction_df['Distance_Motorbike'] >= -100) & (distraction_df['Distance_Motorbike'] <= 100)]
    df_smallcar_control = control_df[(control_df['Distance_SmallCar'] >= -100) & (control_df['Distance_SmallCar'] <= 100)]
    df_bicycle_control = control_df[(control_df['Distance_Bicycle'] >= -100) & (control_df['Distance_Bicycle'] <= 100)]
    df_Motorbike_control = control_df[(control_df['Distance_Motorbike'] >= -100) & (control_df['Distance_Motorbike'] <= 100)]

    # centrlize distance cols:
    min_index = df_smallcar_distraction['Distance_SmallCar'].idxmin()
    df_smallcar_distraction.loc[:min_index-1, 'Distance_SmallCar'] *= -1
    min_index = df_bicycle_distraction['Distance_Bicycle'].idxmin()
    df_bicycle_distraction.loc[:min_index-1, 'Distance_Bicycle'] *= -1
    min_index = df_Motorbike_distraction['Distance_Motorbike'].idxmin()
    df_Motorbike_distraction.loc[:min_index-1, 'Distance_Motorbike'] *= -1
    min_index = df_smallcar_control['Distance_SmallCar'].idxmin()
    df_smallcar_control.loc[:min_index-1, 'Distance_SmallCar'] *= -1
    min_index = df_bicycle_control['Distance_Bicycle'].idxmin()
    df_bicycle_control.loc[:min_index-1, 'Distance_Bicycle'] *= -1
    min_index = df_Motorbike_control['Distance_Motorbike'].idxmin()
    df_Motorbike_control.loc[:min_index-1, 'Distance_Motorbike'] *= -1

    


#    plott(distraction_df,'x_axis','Time_Delta_ms','neg_steer','Fig 1: test run neg_steer over x axis and time')
    #plott(df_bicycle_distraction,'x_axis','Time_Delta_ms','y_delta_bicycle','Fig 2: y_delta_bicycle over location and time')
    #plott(df_Motorbike_distraction,'x_axis','Time_Delta_ms','y_delta_motorbike','Fig 555: y_delta_motorbike over location and time')
    # for presantation:
    ###################### 1. steering input over time/ticks for 2nd run (no distractions)                                                      ######################
    plott(control_df,'x_axis','Time_Delta_ms','neg_steer','Fig 1: control run neg_steer over x axis and time', "[ms]" , "[m]" , "[deg]")
    plott(control_df,'tick','Time_Delta_ms','neg_steer','Fig 2: control run neg_steer over tick and time', "" , "[m]" , "[deg]")

    ###################### 2. steering input over time/ticks for 3nd run (with distractions). verticals on possitions of distractions            ######################
    plott(distraction_df,'x_axis','Time_Delta_ms','neg_steer','Fig 3: distraction run neg_steer over x axis and time', "[ms]" , "[m]" , "[deg]")
    plott(distraction_df,'tick','Time_Delta_ms','neg_steer','Fig 4: distraction run neg_steer over tick and time', "" , "[m]" , "[deg]")

    ###################### 3. first two graphs on same plot - verticals                                                                           ######################
    plot_from_dataframes(control_df, distraction_df, 'x_axis', 'neg_steer',title="Fig 15: control vs distraction run neg_steer over tick and time", labels=('control', 'distraction'))
    #plot_2_plots(control_df,distraction_df,'tick','Time_Delta_ms','neg_steer','Fig 15: control vs distraction run neg_steer over tick and time')
    ###################### 4. best one of distractions 100m before and 100m after - time/ticks and distance to explain indiscrepency verticals   ######################
    #######  small car   ##########
    # steer overtime
    plott(df_smallcar_distraction,'Distance_SmallCar','Time_Delta_ms','neg_steer','Fig 5 :smallcar_distraction steer over Distance_SmallCar and time', "[ms]" , "[m]" , "[deg]")
    plott(df_smallcar_distraction,'tick','Distance_SmallCar','neg_steer','Fig 6 :smallcar_distraction steer over Distance_SmallCar and ticks', "" , "[m]" , "[deg]")

    ########  Bicycle   ##########
    # steer overtime
    plott(df_smallcar_distraction,'Distance_SmallCar','Time_Delta_ms','neg_steer','Fig 7 :smallcar_distraction steer over Distance_SmallCar and time', "[ms]" , "[m]" , "[deg]")
    plott(df_smallcar_distraction,'tick','Distance_SmallCar','neg_steer','Fig 8 :smallcar_distraction steer over Distance_SmallCar and ticks', "" , "[m]" , "[deg]")

    plott(df_bicycle_distraction,'Time_Delta_ms','Distance_Bicycle','neg_steer','Fig 9 :bicycle_distraction steer over Distance_SmallCar and time', "[ms]" , "[m]" , "[deg]")
    plott(df_bicycle_control,'Time_Delta_ms','Distance_Bicycle','neg_steer','Fig 10 :bicycle_control steer over Distance_SmallCar and time', "" , "[m]" , "[deg]")

    ########  Motorbike   ##########
    # steer overtime
    plott(df_Motorbike_distraction,'Time_Delta_ms','Distance_Motorbike','neg_steer','Fig 11 :Motorbike_distraction steer over Distance_SmallCar and time', "[ms]" , "[m]" , "[deg]")
    plott(df_Motorbike_control,'Time_Delta_ms','Distance_Motorbike','neg_steer','Fig 12 : Motorbike_control steer over Distance_SmallCar and time', "" , "[m]" , "[deg]")

    ###################### 5. distractions with yaw - verticals                                                                                     ######################
    plott(distraction_df,'x_axis','Time_Delta_ms','Rotation_Yaw','Fig 16: distraction run Rotation_Yaw over x axis and time', "[ms]" , "[deg]" , "[deg]")
    plott(distraction_df,'tick','Time_Delta_ms','Rotation_Yaw','Fig 17: distraction run Rotation_Yaw over tick and time', "" , "[deg]" , "[deg]")
    plott(control_df,'x_axis','Time_Delta_ms','Rotation_Yaw','Fig 18: control run Rotation_Yaw over x axis and time', "[ms]" , "[deg]" , "[deg]")
    plott(control_df,'tick','Time_Delta_ms','Rotation_Yaw','Fig 19: control run Rotation_Yaw over tick and time', "" , "[deg]" , "[deg]")

    ###################### not a plot - difference in variance when distractions are added.                                                         ######################









    #plot_from_dataframes(control_df, distraction_df, 'Time_Delta_ms', 'Steer',"steer command over time", labels=('A', 'B'))

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
    
    # plot_2_graphs_from_dataframe(control_df, 'Time_Delta_ms', 'Steer','Rotation_Yaw',title="steer and yaw control",y_col_1_label='Steer', y_col_2_label='Rotation_Yaw' )

    # plot_2_graphs_from_dataframe(control_df, 'Time_Delta_ms', 'Steer','Rotation_Yaw',title="steer and yaw control",y_col_1_label='Steer', y_col_2_label='Rotation_Yaw' )
    # plot_2_graphs_with_dual_axes(control_df, 'Time_Delta_ms', 'Steer', 'Rotation_Yaw', title="steer and yaw control with dual axess",y_col_1_label='Steer', y_col_2_label='Rotation_Yaw')
    # plot_2_graphs_with_dual_axes(control_df, 'Time_Delta_ms', 'Steer', 'Rotation_Yaw', title="steer and yaw control with dual axess",y_col_1_label='Steer', y_col_2_label='Rotation_Yaw')
    #plot_2_graphs_with_scaled_axes(control_df, 'Time_Delta_ms', 'Steer','Rotation_Yaw', 'Distance_Bicycle', (-100,100), title="steer and yaw control with dual axess Zoomed", y_col_1_label='Steer', y_col_2_label='Rotation_Yaw')
    # §plot_2_graphs_with_scaled_axes(control_df, 'Distance_Bicycle', 'Steer','Rotation_Yaw', 'Distance_Bicycle', (-100,100), title="steer and yaw control with dual axess Zoomed", y_col_1_label='Steer', y_col_2_label='Rotation_Yaw')

    

    #bicycle_df = control_df[(control_df['Distance_Bicycle'] >= -100) & (df1['Distance_Bicycle'] <= 100)]
    #plot_2_graphs_with_scaled_axes(bicycle_df, 'Time_Delta_ms', 'Steer','Rotation_Yaw', 'Distance_Bicycle', (-100,100), title="steer and yaw control with dual axess Zoomed", y_col_1_label='Steer', y_col_2_label='Rotation_Yaw')

    plt.show()

if __name__ == "__main__":
    main()
