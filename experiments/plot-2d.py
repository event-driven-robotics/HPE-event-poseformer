import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Function to plot skeleton for a given frame
def plot_skeleton(input_path, frame_number):
    # Load the data from the input path
    df = pd.read_csv(input_path)

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
    
    # Get the joint data for the given frame
    frame_data = df[df['frame'] == frame_number]
    
    # If there are no data for the frame, return
    if frame_data.empty:
        print(f"No data for frame {frame_number}")
        return
    
    # Extract x and y coordinates for the joints
    joint_coords = frame_data[['joint', 'x', 'y']].values
    
    # Plot the skeleton
    plt.figure(figsize=(6, 6))
    for edge in skeleton_edges:
        joint1, joint2 = edge
        x1, y1 = joint_coords[joint1][1], joint_coords[joint1][2]
        x2, y2 = joint_coords[joint2][1], joint_coords[joint2][2]
        plt.plot([x1, x2], [y1, y2], 'bo-', alpha=0.7)  # Draw the edges as blue circles and lines
    
    # Plot the joints
    for joint in range(13):
        x, y = joint_coords[joint][1], joint_coords[joint][2]
        plt.scatter(x, y, color='red', zorder=5)  # Red dots for joints

    plt.gca().invert_yaxis()

    # Set labels and title
    plt.title(f'Skeleton for Frame {frame_number}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Plot the skeleton for a given frame in the dataset.")
    parser.add_argument('--input_path', type=str, help="The path to the input CSV file")
    parser.add_argument('--frame', type=int, help="The frame number to plot")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided paths
    plot_skeleton(args.input_path, args.frame)