import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(pred_path, gt_path, plot=False, threshold=150):
        # === Load predictions ===
    predictions = np.load(pred_path, allow_pickle=True)['predictions']

    # === Load ground truth data ===
    ground_truth = np.load(gt_path, allow_pickle=True)['predictions']

    # Check the size of the flattened arrays
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of ground truth: {ground_truth.shape}")

    # Calculate the number of frames based on the total number of elements
    num_frames_pred = predictions.shape[0] // 13  # 13 joints, 3 coordinates
    num_frames_gt = ground_truth.shape[0] // 13  # 13 joints, 3 coordinates
    print(f"Number of frames in predictions: {num_frames_pred}")
    print(f"Number of frames in ground truth: {num_frames_gt}")

    if predictions.shape[0] % 13 != 0 or ground_truth.shape[0] % 13 != 0:
        raise ValueError(f"The flattened data is not divisible by 13.")

    # Reshape predictions and ground truth
    predictions = predictions.reshape(num_frames_pred, 13, 3)
    ground_truth = ground_truth.reshape(num_frames_gt, 13, 3)

    print(f"Reshaped predictions shape: {predictions.shape}")
    print(f"Reshaped ground truth shape: {ground_truth.shape}")
    
    num_frames = min(predictions.shape[0], ground_truth.shape[0])
    predictions = predictions[:num_frames]
    ground_truth = ground_truth[:num_frames]

    # === Apply rotation to align (optional, if needed) ===
    # Assuming you want to align them, you can use rotation like in your example
    theta = np.radians(180)
    rot_z_180 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])

    # Apply rotation to both predictions and ground truth
    predictions_aligned = (predictions - predictions[:, [0], :]) @ rot_z_180.T
    ground_truth_aligned = (ground_truth - ground_truth[:, [0], :]) 

    # === Compute MPJPE ===
    errors = np.linalg.norm(predictions_aligned - ground_truth_aligned, axis=2)  # (num_frames, 13)
    mpjpe = np.mean(errors) 
    print(f"MPJPE: {mpjpe:.2f} mm")

    # === Compute PCK@150mm ===
    correct_keypoints = errors < threshold
    pck = 100.0 * np.sum(correct_keypoints) / correct_keypoints.size
    print(f"PCK @ {threshold:.0f}mm: {pck:.2f}%")

    # === Track MPJPE over time ===
    mpjpe_values = []
    for i in range(num_frames):
        frame_errors = np.linalg.norm(predictions_aligned[i] - ground_truth_aligned[i], axis=1)  # Per joint error for the frame
        mpjpe_frame = np.mean(frame_errors) * 1000  # Convert to mm
        mpjpe_values.append(mpjpe_frame)

    # Plot MPJPE over time if --plot is specified
    if plot:
        plot_mpjpe_over_time(mpjpe_values)

    # === Plot skeleton at a specific frame ===
    frame_idx = 200  # You can change the index to plot a specific frame
    plot_skeleton_3d(predictions_aligned, ground_truth_aligned, frame_idx)

def plot_mpjpe_over_time(mpjpe_values):
    """Plot MPJPE (Mean Per Joint Position Error) over time (frames)."""
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mpjpe_values)), mpjpe_values, color='red', linewidth=2)
    plt.title('MPJPE over Time', fontsize=14)
    plt.xlabel('Frame Index', fontsize=12)
    plt.ylabel('MPJPE (mm)', fontsize=12)
    plt.grid(True)

    plt.xticks(np.arange(0, len(mpjpe_values), 50))  # Add markers every 50 frames

    plt.tight_layout()
    plt.show()

def plot_skeleton_3d(pred_skel, gt_skel, frame_idx=200):
    """Visualize 3D skeletons for a specific frame."""
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select frame
    pred_frame = pred_skel[frame_idx] * 1000  # convert to mm
    gt_frame = gt_skel[frame_idx]
    
    # Define skeleton edges (example)
    skeleton = [
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

    # Plot ground truth (blue)
    for i, j in skeleton:
        ax.plot(
            [gt_frame[i, 0], gt_frame[j, 0]],
            [gt_frame[i, 1], gt_frame[j, 1]],
            [gt_frame[i, 2], gt_frame[j, 2]],
            color='blue', linewidth=2
        )

    # Plot predicted (red dashed)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare 3D pose predictions against ground truth.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the 3D predictions .npz file.")
    parser.add_argument("--gt_path", type=str, required=True, help="Path to the ground truth .npz file.")
    parser.add_argument("--plot", action='store_true', help="Plot MPJPE over time.")
    parser.add_argument("--threshold", type=int, default=150, help="Threshold for PCK (default is 150mm).")

    args = parser.parse_args()
    main(args.pred_path, args.gt_path, plot=args.plot, threshold=args.threshold)
