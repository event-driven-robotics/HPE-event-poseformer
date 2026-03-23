import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Function to import YARP Ground Truth data and convert it to CSV
def import_gt_to_csv(yarp_path, output_path):
    # Ensure yarp_path is a Path object
    yarp_path = Path(yarp_path)

    # Open and read the data.log file
    with open(yarp_path, 'r') as file:
        lines = file.readlines()

    frame_data = []
    for line in lines:
        # Split each line by spaces and extract relevant data
        parts = line.strip().split()
        
        # Extract the time-stamp (first value)
        time_stamp = int(float(parts[0]))  # Convert to int to remove .0

        
        # The joint positions are inside parentheses, so we find the substring within the parentheses
        joint_positions_str = line.split('(')[1].split(')')[0]  # Get the part between '(' and ')'
        
        # Split the joint positions into individual values
        joint_positions = joint_positions_str.split()
        
        # Check if joint_positions contains the correct number of values (26 values for 13 joints)
        if len(joint_positions) != 26:  # 13 joints, each with x and y (13 * 2 = 26)
            print(f"Warning: Incorrect number of joint positions found in line: {line}")
            continue  # Skip this line or handle it differently if necessary

        # Convert joint positions to a list of integers
        joint_positions = [int(val) for val in joint_positions]

        # Now create frame data with joint positions
        for joint_index in range(13):  # Assuming 13 joints in the skeleton
            # Each joint has an x and y coordinate (two values per joint)
            joint_x = joint_positions[joint_index * 2]  # x is at even index
            joint_y = joint_positions[joint_index * 2 + 1]  # y is at odd index
            
            # Append the frame data in the required format
            frame_data.append([time_stamp, joint_index, joint_x, joint_y])

    # Convert the list to a DataFrame
    df_final = pd.DataFrame(frame_data, columns=['frame', 'joint', 'x', 'y'])

    # Save to CSV
    df_final.to_csv(output_path, index=False)
    print(f"Ground truth data saved to {output_path}")

# Main function to handle command-line arguments
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert YARP Ground Truth data to CSV format.")
    parser.add_argument('--input_path', type=str, help="The path to the YARP GT data log file")
    parser.add_argument('--output_path', type=str, help="The path to save the output CSV file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided paths
    import_gt_to_csv(args.input_path, args.output_path)
