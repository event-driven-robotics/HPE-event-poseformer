import argparse
import numpy as np
import matplotlib.pyplot as plt

def main(gt_data_path, frame_idx=0, plot=False):
    # === Load ground truth data ===
    data = np.load(gt_data_path, allow_pickle=True)
    positions_3d = data["positions_3d"].item()

    # === Select test subjects (S9, S11) ===
    subjects_to_evaluate = ["S9", "S11"]
    selected_joint_indices = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

    mpjpe_per_subject = {}

    for subject_name in subjects_to_evaluate:
        subject = positions_3d[subject_name]
        mpjpe_per_action = []
        actions = list(subject.keys())  # Get all actions for the subject
        
        for activity_name in actions:
            activity = subject[activity_name]
            gt_poses = activity  # shape: (N_frames, 32, 3)
            num_frames = gt_poses.shape[0]

            # === Align ground truth and apply 4-frame delay ===
            gt_poses_selected = gt_poses[:, selected_joint_indices, :]

            # Calculate MPJPE with a 4-frame delay
            errors = []
            # skip first 4 frames to avoid index out of range
            for i in range(4, num_frames):  # Skip first 4 frames to avoid index out of range
                # Calculate error between current frame and frame with 4-frame delay
                gt_current = gt_poses_selected[i]
                gt_delayed = gt_poses_selected[i - 4]

                error = np.linalg.norm(gt_current - gt_delayed, axis=1)  # Per joint error
                mpjpe_frame = np.mean(error) * 1000  # Convert to mm
                errors.append(mpjpe_frame)

            # Calculate average MPJPE for this action
            avg_mpjpe = np.mean(errors)
            mpjpe_per_action.append(avg_mpjpe)
            print(f"MPJPE for {subject_name} - {activity_name}: {avg_mpjpe:.2f} mm")

        # Store MPJPE for all actions of the subject
        mpjpe_per_subject[subject_name] = mpjpe_per_action
        print(f"Average MPJPE for {subject_name}: {np.mean(mpjpe_per_action):.2f} mm")

    # === Calculate the overall average MPJPE across all subjects ===
    all_mpjpe_values = [mpjpe for values in mpjpe_per_subject.values() for mpjpe in values]
    overall_avg_mpjpe = np.mean(all_mpjpe_values)
    print(f"Overall Average MPJPE: {overall_avg_mpjpe:.2f} mm")

    # === Plot MPJPE over time for the last action (optional) ===
    if plot:
        mpjpe_values = errors  # MPJPE values for the last action
        plot_mpjpe_over_time(mpjpe_values)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate 3D pose MPJPE with 4-frame delay.")
    parser.add_argument("--gt_data_path", type=str, required=True, help="Path to the ground truth .npz file.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for visualization.")
    parser.add_argument("--plot", action='store_true', help="Plot MPJPE over time.")

    args = parser.parse_args()
    main(args.gt_data_path, args.frame, args.plot)