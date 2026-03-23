import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


SELECTED_JOINT_INDICES = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

SKELETON_EDGES = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9), (9, 10),
    (8, 11), (11, 12), (12, 13),
    (8, 14), (14, 15), (15, 16)
]


def load_predictions(pred_path):
    preds = np.load(pred_path, allow_pickle=True)["predictions"]
    if preds.ndim == 2:
        preds = preds.reshape(preds.shape[0] // 17, 17, 3)
    return preds.astype(np.float32)


def load_gt(subject, activity):
    data = np.load("PoseFormerV2-main/data/data_3d_h36m.npz", allow_pickle=True)
    gt = data["positions_3d"].item()[subject][activity]
    return gt[:, SELECTED_JOINT_INDICES, :].astype(np.float32)


def rot_z(deg):
    t = np.radians(deg)
    return np.array([
        [np.cos(t), -np.sin(t), 0],
        [np.sin(t), np.cos(t), 0],
        [0, 0, 1]
    ], dtype=np.float32)


def root_relative(x):
    return x - x[[0]]


def draw_skeleton(ax, skel, color, label):
    for i, j in SKELETON_EDGES:
        ax.plot(
            [skel[i, 0], skel[j, 0]],
            [skel[i, 1], skel[j, 1]],
            [skel[i, 2], skel[j, 2]],
            color=color,
            linewidth=2
        )
    ax.scatter(
        skel[:, 0], skel[:, 1], skel[:, 2],
        color=color, s=30, label=label
    )


def save_plot(fig, frame_idx, output_dir):
    """ Save the current figure as a PNG image """
    filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    print(f"Saved plot: {filename}")


def main(pred_path_1, pred_path_2, subject, activity, start_frame, end_frame, output_dir):
    preds_1 = load_predictions(pred_path_1)
    preds_2 = load_predictions(pred_path_2)
    gt = load_gt(subject, activity)

    F = min(len(preds_1), len(preds_2), len(gt))
    preds_1 = preds_1[:F]
    preds_2 = preds_2[:F]
    gt = gt[:F]

    R = rot_z(-135)

    gt_aligned = np.array([root_relative(gt[i]) for i in range(F)])
    pred_aligned_1 = np.array([root_relative(preds_1[i]) @ R.T for i in range(F)])
    pred_aligned_2 = np.array([root_relative(preds_2[i]) @ R.T for i in range(F)])

    os.makedirs(output_dir, exist_ok=True)

    for frame_idx in range(start_frame, end_frame):
        idx_m1 = max(0, frame_idx - 1)  # 1-frame delay for prediction
        idx_m4 = max(0, frame_idx - 4)  # 4-frame delay for prediction

        # Fetch the current and delayed frames
        gt_frame = gt_aligned[frame_idx] * 1000
        pred_m1 = pred_aligned_1[idx_m1] * 1000
        pred_m4 = pred_aligned_2[idx_m4] * 1000

        # To introduce variation: Replace pred_m1 with another predicted pose
        # Use a different frame offset for delayed predictions
        delay_frame_m1 = max(0, frame_idx - 1)  # Frame for first delayed prediction
        delay_frame_m2 = max(0, frame_idx - 5)  # Frame for another delayed prediction (new pose)

        pred_m1_new = pred_aligned_1[delay_frame_m1] * 1000
        pred_m2_new = pred_aligned_2[delay_frame_m2] * 1000  # New delayed prediction

        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")

        # Plot the ground truth, first delayed prediction, and second delayed prediction
        draw_skeleton(ax, gt_frame, "blue", f"Ground truth")
        draw_skeleton(ax, pred_m1_new, "red", f"EventPoseFormer")
        draw_skeleton(ax, pred_m2_new, "green", f"HRNet-PoseFormerV2")  # Plot second prediction

        ax.set_title(f"Latency visualization at Frame {frame_idx}", fontsize=13)
        ax.set_xlabel("")  # Remove x-axis label
        ax.set_ylabel("")  # Remove y-axis label
        ax.set_zlabel("")  # Remove z-axis label
        ax.legend()

        # Set plot range
        all_pts = np.concatenate([gt_frame, pred_m1_new, pred_m2_new], axis=0)
        max_range = (all_pts.max(0) - all_pts.min(0)).max() / 2
        mid = all_pts.mean(0)

        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

        ax.view_init(elev=20, azim=-60)
        plt.tight_layout()

        save_plot(fig, frame_idx, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path_1", required=True, help="Path to the first prediction file")
    parser.add_argument("--pred_path_2", required=True, help="Path to the second prediction file")
    parser.add_argument("--subject", required=True)
    parser.add_argument("--activity", required=True)
    parser.add_argument("--start_frame", type=int, required=True)
    parser.add_argument("--end_frame", type=int, required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    main(args.pred_path_1, args.pred_path_2, args.subject, args.activity, args.start_frame, args.end_frame, args.output_dir)
