import cdflib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

cdf_file = cdflib.CDF('demo/output/s11/Greeting.cdf')
gtPose3D = cdf_file['pose']

selected_joint_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]
selected_coord_indices = []
for joint_idx in selected_joint_indices:
    selected_coord_indices.extend([3 * joint_idx, 3 * joint_idx + 1, 3 * joint_idx + 2])

gtPose3D_selected = gtPose3D[:, :, selected_coord_indices]
print(gtPose3D_selected.shape)

data = np.load('demo/output/s11/s11past/predictions_3d.npz')
preds = data['predictions']
print(preds.shape)

preds = preds.reshape((1810, 17, 3))
preds = preds.reshape((1810, 51))
preds = preds[np.newaxis, :, :]

print("Final prediction shape:", preds.shape)
pred_xyz = preds[0].reshape(1810, 17, 3)*1000
gt_xyz = gtPose3D_selected[0].reshape(1808, 17, 3)

pred_aligned = pred_xyz - pred_xyz[:, [0], :]
gt_aligned = gt_xyz - gt_xyz[:, [0], :]

rot_z_neg_135 = np.array([
    [np.cos(np.radians(-135)), -np.sin(np.radians(-135)), 0],
    [np.sin(np.radians(-135)), np.cos(np.radians(-135)), 0],
    [0, 0, 1]
])

pred_aligned = pred_aligned @ rot_z_neg_135.T

min_len = min(pred_aligned.shape[0], gt_aligned.shape[0])
pred_aligned = pred_aligned[:min_len]
gt_aligned = gt_aligned[:min_len]

errors = np.linalg.norm(pred_aligned - gt_aligned, axis=2)  # shape: (2699, 17)

mpjpe = np.mean(errors)  # convert to mm 
print(f"MPJPE: {mpjpe:.2f} mm")

threshold = 150
correct_keypoints = errors < threshold
total_keypoints = errors.size

num_correct = np.sum(correct_keypoints)
pck = 100.0 * num_correct / total_keypoints
print(f"PCK @ {threshold:.0f}mm: {pck:.2f}%")

# ----- 3D Skeleton Visualization -----
def plot_skeleton_3d(pred_skel, gt_skel, frame_idx=0):
    """
    Visualize 3D skeletons for a specific frame.
    pred_skel, gt_skel: shape (N_frames, 17, 3)
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select frame
    pred_frame = pred_skel[frame_idx]
    gt_frame = gt_skel[frame_idx]
    
    # Define skeleton edges
    skeleton = [
        (0, 1), (1, 2), (2, 3),
        (0, 4), (4, 5), (5, 6),
        (0, 7), (7, 8), (8, 9), (9, 10),
        (8, 11), (11, 12), (12, 13),
        (8, 14), (14, 15), (15, 16)
    ]

    # ---- Plot ground truth (blue) ----
    for i, j in skeleton:
        ax.plot(
            [gt_frame[i, 0], gt_frame[j, 0]],
            [gt_frame[i, 1], gt_frame[j, 1]],
            [gt_frame[i, 2], gt_frame[j, 2]],
            color='blue', linewidth=2
        )

    # ---- Plot predicted (red dashed) ----
    for i, j in skeleton:
        ax.plot(
            [pred_frame[i, 0], pred_frame[j, 0]],
            [pred_frame[i, 1], pred_frame[j, 1]],
            [pred_frame[i, 2], pred_frame[j, 2]],
            color='red', linestyle='--', linewidth=2
        )

    # Plot joint points
    gt_scatter = ax.scatter(gt_frame[:, 0], gt_frame[:, 1], gt_frame[:, 2],
                            color='blue', s=30, label='Ground Truth')
    pred_scatter = ax.scatter(pred_frame[:, 0], pred_frame[:, 1], pred_frame[:, 2],
                              color='red', s=30, label='Prediction')

    # Labels and styling
    ax.set_title(f"3D Skeleton Comparison (Frame {frame_idx})", fontsize=13)
    ax.legend(handles=[gt_scatter, pred_scatter])
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    # Equal aspect ratio
    all_points = np.concatenate([pred_frame, gt_frame], axis=0)
    max_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2.0
    mid = all_points.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()


# heights = []
# skeleton_data = gt_xyz  # or gt_aligned
# right_ankle_idx = 3
# left_ankle_idx = 6
# head_idx = 10
# for f in range(skeleton_data.shape[0]):
#     foot_y = np.mean([
#         skeleton_data[f, right_ankle_idx, 2],
#         skeleton_data[f, left_ankle_idx, 2]
#     ])
#     head_y = skeleton_data[f, head_idx, 2]
#     heights.append(abs(head_y - foot_y))

# avg_height_mm = np.mean(heights)
# print(f"Average skeleton height: {avg_height_mm:.1f} mm ({avg_height_mm/1000:.2f} m)")
# choose which skeleton to measure
skeleton_data = gt_aligned  # or pred_aligned

# indices
head_idx = 10
right_ankle_idx = 3
left_ankle_idx = 6

# compute mean ankle height (in mm)
foot_y = np.mean([
    skeleton_data[1000, right_ankle_idx, 2],
    skeleton_data[1000, left_ankle_idx, 2]
])
head_y = skeleton_data[1000, head_idx, 2]

# height in mm and meters
height_mm = abs(head_y - foot_y)
height_m = height_mm / 1000

print(f"Estimated skeleton height: {height_mm:.1f} mm ({height_m:.2f} m)")



# ---- Call it for a given frame index ----
plot_skeleton_3d(pred_aligned, gt_aligned, frame_idx=1000)

# print("gt:", gt_xyz[200])
# print("pred:", pred_xyz[103])