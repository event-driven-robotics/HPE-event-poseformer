import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the prediction and ground truth data
predictions = np.load('../outputs/cam2_S1_Directions/predictions_3d.npz', allow_pickle=True)['predictions']  # shape: (N_frames * 17, 3)
# Load the ground truth data (adjust path if needed)
data = np.load("data/data_3d_h36m.npz", allow_pickle=True)
positions_3d = data["positions_3d"].item()

# Extract GT data for subject "S1" and activity "Directions"
subject = positions_3d["S1"]
activity = subject["Directions"]  # shape: (N_frames, 32, 3)
seq = activity  # Selecting the first activity sequence (adjust if needed)
gt_poses_selected = seq  # shape: (N_frames, 32, 3)

# Selected joint indices (17 joints)
selected_joint_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# Define the rotation angle (-135 degrees) and convert to radians
theta = -135 * np.pi / 180  # Convert -135 degrees to radians

# Rotation matrix around Z-axis for -135 degrees
rotation_matrix = np.array([
    [np.cos(theta), -np.sin(theta), 0],
    [np.sin(theta), np.cos(theta), 0],
    [0, 0, 1]
])

# Check the shape of predictions
print("Shape of predictions before reshaping:", predictions.shape)  # Should be (N_frames * 17, 3)

# Reshape the predictions to (N_frames, 17, 3) if it's currently (N_frames * 17, 3)
if predictions.ndim == 2 and predictions.shape[1] == 3:
    num_frames = predictions.shape[0] // 17  # Calculate the number of frames
    predictions = predictions.reshape(num_frames, 17, 3)

# Verify the reshaped predictions shape
print("Shape of predictions after reshaping:", predictions.shape)  # Should be (N_frames, 17, 3)

# Determine the number of frames to process (use the smaller length)
num_frames = min(predictions.shape[0], gt_poses_selected.shape[0])

# Initialize list to store MPJPE for each frame
mpjpe_list = []

# Loop over all frames (up to the number of frames in both predictions and ground truth)
for frame_idx in range(num_frames):
    # Extract the 3D positions for the current frame
    pred_frame = predictions[frame_idx]  # shape: (17, 3)
    gt_frame = gt_poses_selected[frame_idx]  # shape: (32, 3)

    # Apply the rotation to the predicted joints
    pred_frame_rotated = np.dot(pred_frame - pred_frame[0], rotation_matrix.T)  # Rotate the predicted joints by -135 degrees
    pred_frame_rotated += pred_frame[0]  # Translate back to original position (centering at the hip)

    # Select the relevant 17 joints from the ground truth (using the provided indices)
    gt_frame_selected = gt_frame[selected_joint_indices]  # shape: (17, 3)

    # If predictions and ground truth are incorrectly shaped, reshape them
    if pred_frame_rotated.ndim == 1:
        pred_frame_rotated = pred_frame_rotated.reshape(17, 3)  # Reshape if needed

    if gt_frame_selected.ndim == 1:
        gt_frame_selected = gt_frame_selected.reshape(17, 3)  # Reshape if needed

    # Translation Normalization: Set the first joint (hip) to (0, 0, 0)
    pred_frame_normalized = pred_frame_rotated - pred_frame_rotated[0]  # Subtract position of first joint (hip)
    gt_frame_normalized = gt_frame_selected - gt_frame_selected[0]  # Subtract position of first joint (hip)

    # Calculate the Euclidean distance between each corresponding joint in the prediction and ground truth
    distances = np.linalg.norm(pred_frame_normalized - gt_frame_normalized, axis=1)  # shape: (17,)

    # Calculate the mean error for the current frame (MPJPE for this frame)
    mpjpe_frame = np.mean(distances)
    mpjpe_list.append(mpjpe_frame)

# Calculate the average MPJPE across all frames
mpjpe_all_frames = np.mean(mpjpe_list) * 1000  # Convert to mm

print(f"Mean Per Joint Position Error (MPJPE) across all frames: {mpjpe_all_frames:.2f}")

# Choose a frame to visualize (e.g., the first frame)
frame_idx = 100

# Extract the 3D positions for the selected frame (first frame)
pred_frame = predictions[frame_idx] * 1000  # shape: (17, 3)
gt_frame = gt_poses_selected[frame_idx] * 1000  # shape: (32, 3)

# Apply the rotation to the predicted joints
pred_frame_rotated = np.dot(pred_frame - pred_frame[0], rotation_matrix.T)  # Rotate the predicted joints by -135 degrees
pred_frame_rotated += pred_frame[0]  # Translate back to original position (centering at the hip)

# Select the relevant 17 joints from the ground truth
gt_frame_selected = gt_frame[selected_joint_indices]  # shape: (17, 3)

# If predictions and ground truth are incorrectly shaped, reshape them:
if pred_frame_rotated.ndim == 1:
    pred_frame_rotated = pred_frame_rotated.reshape(17, 3)  # Reshape if needed

if gt_frame_selected.ndim == 1:
    gt_frame_selected = gt_frame_selected.reshape(17, 3)  # Reshape if needed

# Translation Normalization: Set the first joint (hip) to (0, 0, 0)
pred_frame_normalized = pred_frame_rotated - pred_frame_rotated[0]  # Subtract position of first joint (hip)
gt_frame_normalized = gt_frame_selected - gt_frame_selected[0]  # Subtract position of first joint (hip)

# Create the plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the normalized predicted joints (in blue)
ax.scatter(pred_frame_normalized[:, 0], pred_frame_normalized[:, 1], pred_frame_normalized[:, 2], color='b', label='Prediction', s=100)

# Plot the normalized ground truth joints (in red)
ax.scatter(gt_frame_normalized[:, 0], gt_frame_normalized[:, 1], gt_frame_normalized[:, 2], color='r', label='Ground Truth', s=100)

# Skeleton connecting the joints
skeleton = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]

# Connect the joints as per the skeleton (predictions)
for start, end in skeleton:
    ax.plot([pred_frame_normalized[start, 0], pred_frame_normalized[end, 0]],
            [pred_frame_normalized[start, 1], pred_frame_normalized[end, 1]],
            [pred_frame_normalized[start, 2], pred_frame_normalized[end, 2]], color='b', linewidth=2)

# Connect the joints as per the skeleton (ground truth)
for start, end in skeleton:
    ax.plot([gt_frame_normalized[start, 0], gt_frame_normalized[end, 0]],
            [gt_frame_normalized[start, 1], gt_frame_normalized[end, 1]],
            [gt_frame_normalized[start, 2], gt_frame_normalized[end, 2]], color='r', linewidth=2)

# Add labels for the joints (optional)
for i in range(len(pred_frame)):
    ax.text(pred_frame_normalized[i, 0], pred_frame_normalized[i, 1], pred_frame_normalized[i, 2], str(i), color='b', fontsize=12)
    ax.text(gt_frame_normalized[i, 0], gt_frame_normalized[i, 1], gt_frame_normalized[i, 2], str(i), color='r', fontsize=12)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the title
ax.set_title(f'Frame {frame_idx} - Predicted vs Ground Truth')

# Show legend
ax.legend()

# Show the plot
plt.show()