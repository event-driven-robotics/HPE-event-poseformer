import numpy as np
import matplotlib.pyplot as plt

GT_PATH = "PoseFormerV2-main/data/data_2d_h36m_cpn_ft_h36m_dbb.npz"
PRED_PATH = "outputs/skip/Cam2_S11_Greeting/keypoints_17.npz"

SUBJECT = "S11"
ACTION = "Greeting"
CAM_IDX = 1    

FPS = 50.0

HIP_INDEX = 0

SKELETON_17 = [
    (0, 1), (1, 2), (2, 3),          
    (1, 4), (4, 5), (5, 6),          
    (1, 7), (7, 8), (8, 9), (9, 10), 
    (7, 11), (11, 12), (12, 13),     
    (7, 14), (14, 15), (15, 16)     
]


def load_npz_files(gt_path, pred_path):
    gt_npz = np.load(gt_path, allow_pickle=True)
    pred_npz = np.load(pred_path, allow_pickle=True)
    positions_2d = gt_npz["positions_2d"].item()
    keypoints = pred_npz["keypoints"]
    return positions_2d, keypoints


def get_gt_array(positions_2d, subject="S6", action="Directions 1", cam_idx=0):
    gt_entry = positions_2d[subject][action]
    if isinstance(gt_entry, list):
        return gt_entry[cam_idx]
    return gt_entry


def root_center_skeletons(gt, preds, hip_idx=HIP_INDEX):
    """
    Root-center GT and preds by subtracting the hip joint from all joints
    in each frame.
    gt, preds: (T, J, 2)
    """
    # (T, 1, 2) hip positions
    gt_hip = gt[:, hip_idx:hip_idx+1, :]
    pred_hip = preds[:, hip_idx:hip_idx+1, :]

    gt_centered = gt - gt_hip
    preds_centered = preds - pred_hip

    return gt_centered, preds_centered


def compute_mpjpe(positions_2d, keypoints,
                  subject="S6", action="Directions 1",
                  cam_idx=0, hip_idx=HIP_INDEX):
    """
    Compute MPJPE (Mean Per Joint Position Error) in 2D (mm),
    after:
      - temporal alignment (GT trimmed from beginning if longer)
      - joint mapping (32->17 if needed)
      - root-centering by hip for both GT and preds
    """

    gt_all = get_gt_array(positions_2d, subject, action, cam_idx)  # (T_gt, J_gt, 2)
    T_gt, J_gt, _ = gt_all.shape

    preds = keypoints[0] if keypoints.ndim == 4 else keypoints   # (T_pred, J_pred, 2)
    T_pred, J_pred, _ = preds.shape

    # Joint mapping
    if J_gt > J_pred:
        sel_idx = SELECTED_JOINT_INDICES_32_TO_17
        if max(sel_idx) >= J_gt:
            raise ValueError(
                f"Selected joint index {max(sel_idx)} out of bounds for GT with {J_gt} joints."
            )
        gt = gt_all[:, sel_idx, :]      # (T_gt, 17, 2)
    elif J_gt == J_pred:
        gt = gt_all                     # (T_gt, 17, 2)
    else:
        raise ValueError(
            f"GT has fewer joints ({J_gt}) than predictions ({J_pred}). "
            "Check your data layout."
        )

    # If GT is longer keep last T_pred frames of GT
    if T_gt > T_pred:
        gt = gt[T_gt - T_pred:]   # last T_pred frames
        preds = preds             # unchanged
        T = T_pred
    else:
        # otherwise trim both to the shortest
        T = min(T_gt, T_pred)
        gt = gt[:T]
        preds = preds[:T]

    # Now shapes: (T, 17, 2)
    assert gt.shape == preds.shape, f"Shape mismatch after alignment: {gt.shape} vs {preds.shape}"

    # ----- ROOT-CENTERING ON HIP -----
    gt_centered, preds_centered = root_center_skeletons(gt, preds, hip_idx=hip_idx)

    # Per-joint Euclidean error: (T, 17)
    errors = np.linalg.norm(preds_centered - gt_centered, axis=-1)

    # MPJPE per frame: (T,)
    mpjpe_per_frame = errors.mean(axis=-1)

    # Overall MPJPE: scalar
    overall_mpjpe = mpjpe_per_frame.mean()

    return overall_mpjpe, mpjpe_per_frame, gt_centered, preds_centered

def plot_mpjpe_over_frames(mpjpe_per_frame, title="2D MPJPE over Frames"):
    """
    Plot MPJPE per frame, X-axis = frame index (0..T-1)
    """
    T = len(mpjpe_per_frame)
    x = np.arange(T)   # frame indices

    plt.figure(figsize=(10, 4))
    plt.plot(x, mpjpe_per_frame, linewidth=1.5)
    # bigger font on the axis labels and title for better readability
    plt.xticks(np.arange(0, len(mpjpe_per_frame), 500), fontsize=14)  # Add markers every 500 frames
    plt.yticks(fontsize=14)
    plt.xlabel("Frame Index", fontsize=16)
    plt.ylabel("MPJPE (mm)", fontsize=16)
    plt.grid(True)
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_skeleton_frame(gt_frame_centered, pred_frame_centered,
                        skeleton=SKELETON_17,
                        show_indices=False,
                        title="GT (green) vs Pred (red), hip-centered"):
    """
    gt_frame_centered, pred_frame_centered: (17, 2)
    Assumes already root-centered (hip at (0,0)).
    """
    gt = gt_frame_centered
    pred = pred_frame_centered

    plt.figure(figsize=(6, 6))

    # GT skeleton (green)
    for (i, j) in skeleton:
        plt.plot([gt[i, 0], gt[j, 0]],
                 [gt[i, 1], gt[j, 1]],
                 color="green", linewidth=2)

    # Pred skeleton (red)
    for (i, j) in skeleton:
        plt.plot([pred[i, 0], pred[j, 0]],
                 [pred[i, 1], pred[j, 1]],
                 color="red", linewidth=2, linestyle="--")

    # Joints
    plt.scatter(gt[:, 0], gt[:, 1], c="green", s=40, label="GT")
    plt.scatter(pred[:, 0], pred[:, 1], c="red", s=40, label="Pred")

    if show_indices:
        for idx in range(gt.shape[0]):
            plt.text(gt[idx, 0], gt[idx, 1], str(idx), color="green")
            plt.text(pred[idx, 0], pred[idx, 1], str(idx), color="red")

    plt.gca().invert_yaxis()  
    plt.legend()
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    positions_2d, keypoints = load_npz_files(GT_PATH, PRED_PATH)

    print("GT sample shape:", get_gt_array(positions_2d, SUBJECT, ACTION, CAM_IDX).shape)
    print("Pred shape:", keypoints.shape)

    overall_mpjpe, mpjpe_per_frame, gt_centered, preds_centered = compute_mpjpe(
        positions_2d,
        keypoints,
        subject=SUBJECT,
        action=ACTION,
        cam_idx=CAM_IDX,
        hip_idx=HIP_INDEX
    )

    print("\n--- RESULTS (hip-normalized) ---")
    print(f"Overall MPJPE: {overall_mpjpe:.4f} mm")

    plot_mpjpe_over_frames(
        mpjpe_per_frame,
        title=f"2D MPJPE over Time"
    )

    frame = 100 
    plot_skeleton_frame(
        gt_centered[frame],
        preds_centered[frame],
        show_indices=True,
        title=f"Hip-centered skeletons @ frame {frame}"
    )
