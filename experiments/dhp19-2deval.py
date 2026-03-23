import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Skeleton edges (as per the provided list)
skeleton_edges = [
    (0, 1),   # head → shoulder_right
    (0, 2),   # head → shoulder_left
    (1, 2),   # shoulder_right ↔ shoulder_left
    (1, 3),   # right arm
    (3, 7),
    (2, 4),   # left arm
    (4, 8),
    (1, 6),   # shoulders → hips
    (2, 5),
    (5, 6),
    (6, 9),   # right leg
    (9, 11),
    (5, 10),  # left leg
    (10, 12)
]

# Function to calculate MPJPE (Mean Per Joint Position Error)
def calculate_mpjpe(gt_coords, pred_coords):
    distances = np.linalg.norm(gt_coords - pred_coords, axis=1)  # Calculate Euclidean distances between GT and Pred
    return np.mean(distances)  # Return the average distance (MPJPE)

# Function to calculate PCK@150mm (Percentage of Correct Keypoints)
def calculate_pck(gt_coords, pred_coords, threshold=150):
    distances = np.linalg.norm(gt_coords - pred_coords, axis=1)  # Calculate Euclidean distances between GT and Pred
    correct_joints = np.sum(distances <= threshold)  # Count joints within the threshold
    return correct_joints / len(gt_coords)  # Percentage of correctly predicted joints

# Function to plot skeleton for a given frame
def plot_skeleton(gt_input_path, pred_input_path, frame_number):
    # Load the GT and prediction data from the input paths
    gt_df = pd.read_csv(gt_input_path)
    pred_df = pd.read_csv(pred_input_path)
    print(f"Plotting skeleton for frame {frame_number}...")
    # Get the joint data for the given frame (GT and prediction)
    gt_frame_data = gt_df[gt_df['frame'] == frame_number]
    pred_frame_data = pred_df[pred_df['frame'] == frame_number]
    
    # If there are no data for the frame, return
    if gt_frame_data.empty or pred_frame_data.empty:
        print(f"No data for frame {frame_number}")
        return
    
    # Extract x and y coordinates for the joints (GT and prediction)
    gt_joint_coords = gt_frame_data[['joint', 'x', 'y']].values[:, 1:]
    pred_joint_coords = pred_frame_data[['joint', 'x', 'y']].values[:, 1:]
    
    # Plot the skeletons (GT in blue, prediction in red)
    plt.figure(figsize=(6, 6))
    
    # Plot the skeleton edges for GT
    for edge in skeleton_edges:
        joint1, joint2 = edge
        x1, y1 = gt_joint_coords[joint1]
        x2, y2 = gt_joint_coords[joint2]
        plt.plot([x1, x2], [y1, y2], 'bo-', alpha=0.7)  # Blue for GT
    
    # Plot the joints for GT
    for joint in range(13):
        x, y = gt_joint_coords[joint]
        plt.scatter(x, y, color='blue', zorder=5)  # Blue for GT joints
    
    # Plot the skeleton edges for predicted (in red)
    for edge in skeleton_edges:
        joint1, joint2 = edge
        x1, y1 = pred_joint_coords[joint1]
        x2, y2 = pred_joint_coords[joint2]
        plt.plot([x1, x2], [y1, y2], 'ro-', alpha=0.7)  # Red for predicted
    
    # Plot the joints for predicted
    for joint in range(13):
        x, y = pred_joint_coords[joint]
        plt.scatter(x, y, color='red', zorder=5)  # Red for predicted joints
    
    plt.gca().invert_yaxis()  # Invert y-axis (optional depending on the origin)
    plt.title(f'Skeleton for Frame {frame_number}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.legend(['GT Skeleton', 'Predicted Skeleton'])
    plt.show()

# Main function to handle command-line arguments
def compare_and_visualize(gt_input_path, pred_input_path, frame):
    # Load the GT and prediction data from the input paths
    gt_df = pd.read_csv(gt_input_path)
    pred_df = pd.read_csv(pred_input_path)
    
    # Check if the number of frames match
    if len(gt_df['frame'].unique()) != len(pred_df['frame'].unique()):
        print("Warning: The number of frames in GT and Pred files do not match.")
    
    # Initialize lists to store MPJPE and PCK for all frames
    mpjpe_list = []
    pck_list = []
    
    # Loop through all unique frames in GT data (assuming frames are consistent in both GT and Pred)
    all_frame_numbers = sorted(gt_df['frame'].unique())
    
    for frame_number in all_frame_numbers:
        # Get the joint data for the given frame (GT and prediction)
        gt_frame_data = gt_df[gt_df['frame'] == frame_number]
        pred_frame_data = pred_df[pred_df['frame'] == frame_number]
        
        # If there are no data for the frame, skip it
        if gt_frame_data.empty or pred_frame_data.empty:
            print(f"Skipping frame {frame_number} as no data available for this frame in GT or Pred.")
            continue
        
        # Extract x and y coordinates for the joints (GT and prediction)
        gt_joint_coords = gt_frame_data[['joint', 'x', 'y']].values[:, 1:]
        pred_joint_coords = pred_frame_data[['joint', 'x', 'y']].values[:, 1:]
        
        # Calculate MPJPE for this frame
        mpjpe = calculate_mpjpe(gt_joint_coords, pred_joint_coords)
        mpjpe_list.append(mpjpe)
        
        # Calculate PCK@150mm for this frame
        pck = calculate_pck(gt_joint_coords, pred_joint_coords, threshold=150)
        pck_list.append(pck)
    
    # Calculate the average MPJPE and PCK@150mm for all frames
    avg_mpjpe = np.mean(mpjpe_list) if mpjpe_list else 0
    avg_pck = np.mean(pck_list) if pck_list else 0
    
    # Print the results
    print(f"Average MPJPE over all frames: {avg_mpjpe:.2f} mm")
    print(f"Average PCK@150mm over all frames: {avg_pck * 100:.2f}%")
    
    # If a specific frame was requested, plot that frame's skeletons
    if frame is not None:
        plot_skeleton(gt_input_path, pred_input_path, frame)

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Compare and visualize skeletons for GT and Pred datasets.")
    parser.add_argument('--gt_input_path', type=str, help="The path to the ground truth CSV file")
    parser.add_argument('--pred_input_path', type=str, help="The path to the predicted CSV file")
    parser.add_argument('--frame', type=int, help="The frame number to visualize (optional)")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided paths
    compare_and_visualize(args.gt_input_path, args.pred_input_path, args.frame)
