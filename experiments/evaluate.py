import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def evaluate_single(pred_path, positions_3d, subject_name, activity_name, frame_idx=0, delay=0, plot=False, silent=False):
    # === Load predictions ===
    if not os.path.exists(pred_path):
        if not silent:
            print(f"Skipping: {pred_path} (not found)")
        return None, None

    predictions_data = np.load(pred_path, allow_pickle=True)
    if 'predictions' not in predictions_data:
        if not silent:
            print(f"Skipping: {pred_path} (no 'predictions' key)")
        return None, None
    predictions = predictions_data['predictions']

    # === Load ground truth data ===
    if subject_name not in positions_3d:
        if not silent:
            print(f"Skipping: Subject {subject_name} not in GT data")
        return None, None
    
    subject = positions_3d[subject_name]
    if activity_name not in subject:
        # Try some common variations if not found exactly
        alt_activity = activity_name.replace("_", " ")
        if alt_activity in subject:
            activity_name = alt_activity
        else:
            if not silent:
                print(f"Skipping: Activity {activity_name} not in subject {subject_name}")
                print(f"Available activities: {list(subject.keys())}")
            return None, None
    
    gt_poses = subject[activity_name]  # shape: (N_frames, 32, 3)

    # === Select the 17 joints you used ===
    selected_joint_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    # === Rotation matrix (-135° around Z) ===
    theta = np.radians(-135)
    rot_z_neg_135 = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ])

    # === Reshape predictions ===
    if predictions.ndim == 2 and predictions.shape[1] == 3:
        num_frames_raw = predictions.shape[0] // 17
        predictions = predictions.reshape(num_frames_raw, 17, 3)
    
    fps_pred = 50
    fps_gt = 50

    predictions, gt_poses = match_fps(predictions, gt_poses, fps_pred, fps_gt)
    
    if not silent:
        print(f"Evaluating {subject_name} - {activity_name} (Delay: {delay} frames)")
        print(" Predictions shape:", predictions.shape)
        print(" GT shape:", gt_poses.shape)

    # === Align and compare ===
    num_frames = min(predictions.shape[0], gt_poses.shape[0])
    
    # Compute start indices so we trim from the beginning of the longer sequence
    start_gt = max(0, gt_poses.shape[0] - num_frames)
    start_pred = max(0, predictions.shape[0] - num_frames)

    # Apply additional delay if requested
    # Positive delay: prediction[i] compared with GT[i + delay]
    if delay > 0:
        if num_frames > delay:
            start_gt += delay
            num_frames -= delay
        else:
            if not silent: print(f" Error: Delay {delay} exceeds segment length {num_frames}")
            return None, None
    elif delay < 0:
        abs_delay = abs(delay)
        if num_frames > abs_delay:
            start_pred += abs_delay
            num_frames -= abs_delay
        else:
            if not silent: print(f" Error: Delay {delay} exceeds segment length {num_frames}")
            return None, None

    gt_poses_selected = gt_poses[start_gt : start_gt + num_frames, selected_joint_indices, :]
    pred_xyz = predictions[start_pred : start_pred + num_frames]

    # Apply rotation
    pred_aligned = (pred_xyz - pred_xyz[:, [0], :]) @ rot_z_neg_135.T
    gt_aligned = gt_poses_selected - gt_poses_selected[:, [0], :]

    # === Compute MPJPE ===
    errors = np.linalg.norm(pred_aligned - gt_aligned, axis=2)  # (frames, 17)
    mpjpe = np.mean(errors) * 1000  # mm
    
    # === Compute PCK@150mm ===
    threshold = 150  # mm
    correct_keypoints = errors * 1000 < threshold
    pck = 100.0 * np.sum(correct_keypoints) / correct_keypoints.size
    
    if not silent:
        print(f" MPJPE: {mpjpe:.2f} mm")
        print(f" PCK @ {threshold:.0f}mm: {pck:.2f}%")
    
    # === Track MPJPE over time ===
    mpjpe_values = []
    if plot or not silent:
        for i in range(num_frames):
            frame_errors = np.linalg.norm(pred_aligned[i] - gt_aligned[i], axis=1)
            mpjpe_frame = np.mean(frame_errors) * 1000
            mpjpe_values.append(mpjpe_frame)

    if plot:
        plot_mpjpe_over_time(mpjpe_values)
        plot_skeleton_3d(pred_aligned, gt_aligned, frame_idx)

    return mpjpe, pck

def main(pred_path, subject_name=None, activity_name=None, frame_idx=0, delay=0, plot=False):
    # === Load ground truth data ===
    print("Loading GT data...")
    data = np.load("PoseFormerV2-main/data/data_3d_h36m.npz", allow_pickle=True)
    positions_3d = data["positions_3d"].item()

    results = []

    if os.path.isdir(pred_path):
        print(f"Processing directory: {pred_path}")
        folders = [f for f in os.listdir(pred_path) if os.path.isdir(os.path.join(pred_path, f))]
        folders.sort()
        
        for folder in folders:
            # Expected format: cam2_S11_Directions or cam2_S11_Directions_1
            parts = folder.split('_')
            if len(parts) >= 3:
                # parts[0] is usually camX
                s_name = parts[1]
                # activity might have sub-IDs
                a_name = " ".join(parts[2:])
                
                folder_path = os.path.join(pred_path, folder, "predictions_3d.npz")
                if os.path.exists(folder_path):
                    mpjpe, pck = evaluate_single(folder_path, positions_3d, s_name, a_name, frame_idx, delay, plot=False, silent=False)
                    if mpjpe is not None:
                        results.append({
                            'folder': folder,
                            'subject': s_name,
                            'activity': a_name,
                            'mpjpe': mpjpe,
                            'pck': pck
                        })
                        print("-" * 30)
            else:
                print(f"Skipping folder with unexpected name format: {folder}")

        if results:
            print("\n" + "="*50)
            print(f"{'Folder':<40} | {'MPJPE (mm)':<12} | {'PCK (%)':<8}")
            print("-" * 65)
            all_mpjpes = [r['mpjpe'] for r in results]
            all_pcks = [r['pck'] for r in results]
            
            for r in results:
                print(f"{r['folder']:<40} | {r['mpjpe']:>12.2f} | {r['pck']:>8.2f}")
            
            print("-" * 65)
            print(f"{'AVERAGE':<40} | {np.mean(all_mpjpes):>12.2f} | {np.mean(all_pcks):>8.2f}")
            print("="*50)
        else:
            print("No valid results found in directory.")

    else:
        # Single file evaluation
        if not subject_name or not activity_name:
            # Try to infer from path if not provided
            parent_folder = os.path.basename(os.path.dirname(os.path.abspath(pred_path)))
            parts = parent_folder.split('_')
            if len(parts) >= 3:
                subject_name = parts[1]
                activity_name = " ".join(parts[2:])
            else:
                print("Error: For single file evaluation, --subject and --activity must be provided or path must follow naming convention.")
                return

        evaluate_single(pred_path, positions_3d, subject_name, activity_name, frame_idx, delay, plot, silent=False)

    
def match_fps(predictions, gt_poses, fps_pred, fps_gt):
    """f
    Downsample the higher-FPS sequence so both run at the same FPS.
    Assumes integer ratio (e.g., 50 -> 25, 60 -> 30, etc.).
    """
    if fps_pred == fps_gt:
        return predictions, gt_poses

    if fps_gt > fps_pred:
        ratio = fps_gt / fps_pred
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("Non-integer FPS ratio, need interpolation.")
        step = int(round(ratio))
        gt_poses = gt_poses[::step]
    else:
        ratio = fps_pred / fps_gt
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("Non-integer FPS ratio, need interpolation.")
        step = int(round(ratio))
        predictions = predictions[::step]

    return predictions, gt_poses

    
def plot_mpjpe_over_time(mpjpe_values):
    """
    Plot MPJPE (Mean Per Joint Position Error) over time (frames).
    mpjpe_values: List of MPJPE values for each frame.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(len(mpjpe_values)), mpjpe_values, color='red', linewidth=2)
    plt.title('MPJPE over Time', fontsize=14)
    plt.xlabel('Frame Index', fontsize=14)
    plt.ylabel('MPJPE (mm)', fontsize=14)
    plt.grid(True)

    plt.xticks(np.arange(0, len(mpjpe_values), 50), fontsize=14)  # Add markers every 50 frames
    plt.yticks(fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_skeleton_3d(pred_skel, gt_skel, frame_idx=0):
    """
    Visualize 3D skeletons for a specific frame.
    pred_skel, gt_skel: shape (N_frames, 17, 3)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select frame
    pred_frame = pred_skel[frame_idx] * 1000  # convert to mm
    gt_frame = gt_skel[frame_idx] * 1000
    
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

# def determine_height(skeleton_data, frame_idx=0):
#     """
#     Estimate skeleton height from 3D joint data.
#     skeleton_data: shape (N_frames, 17, 3)
#     """
#     # indices
#     head_idx = 10
#     right_ankle_idx = 3
#     left_ankle_idx = 6

#     # compute mean ankle height (in m)
#     foot_y = np.mean([
#         skeleton_data[frame_idx, right_ankle_idx, 2],
#         skeleton_data[frame_idx, left_ankle_idx, 2]
#         ])
#     head_y = skeleton_data[frame_idx, head_idx, 2]

#     # height in meters and centimeters
#     height_m = abs(head_y - foot_y)
#     height_cm = height_m * 100

#     print(f"Estimated skeleton height: {height_m:.2f} m ({height_cm:.2f} cm)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D pose predictions against Human3.6M ground truth.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to the 3D predictions .npz file.")
    parser.add_argument("--subject", type=str, help="Subject ID (e.g. S1, S5, S11).")
    parser.add_argument("--activity", type=str, help="Activity name (e.g. Directions, Greeting).")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for visualization and height estimation.")
    parser.add_argument("--delay", type=int, default=0, help="Delay in frames for comparison (e.g. 4 means pred[0] vs GT[4]).")
    parser.add_argument("--plot", action='store_true', help="Plot MPJPE over time.")

    args = parser.parse_args()
    main(args.pred_path, args.subject, args.activity, args.frame, args.delay, args.plot)