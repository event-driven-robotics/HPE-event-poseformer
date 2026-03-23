import pandas as pd
import argparse

def process_motion_data(input_path, output_path):
    # Load the dataset and split the columns by space
    df_split = pd.read_csv(input_path, header=None, delimiter=" ")

    # Assign column names (timestamp, delay, followed by alternating x, y for each joint)
    columns = []
    for i in range(1, 14):
        columns.append(f'joint_{i}_x')
        columns.append(f'joint_{i}_y')

    df_split.columns = ['timestamp', 'delay'] + columns

    # Sort the dataframe by timestamp
    df_sorted = df_split.sort_values(by='timestamp').reset_index(drop=True)

    # Create a new dataframe to store Frame, Joint, x, y
    frame_data = []

    # Loop through the rows and extract frame number, joint index, x, and y values
    for frame_number, row in df_sorted.iterrows():
        for joint in range(13):  # 13 joints, indexing from 0
            # Extract x and y for the current joint
            joint_x = row[f'joint_{joint + 1}_x']  # The x value for the joint
            joint_y = row[f'joint_{joint + 1}_y']  # The y value for the joint

            # Append to frame_data with the correct format
            frame_data.append([frame_number, joint, joint_x, joint_y])

    # Convert the data into a new DataFrame
    df_final = pd.DataFrame(frame_data, columns=['frame', 'joint', 'x', 'y'])

    # Save the processed data to the output path
    df_final.to_csv(output_path, index=False)

    # Output the file path
    print(f"File saved to {output_path}")

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process motion data and convert it to a specific format.")
    parser.add_argument('--input_path', type=str, help="The path to the input CSV file")
    parser.add_argument('--output_path', type=str, help="The path to the output CSV file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided paths
    process_motion_data(args.input_path, args.output_path)
