import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

GT_CODES = ["HEDO","RCLO","LCLO","RHUO","LHUO","LFEP","RFEP","RRAO","LRAO","RFEO","LFEO","RTIO","LTIO"]

def load_gt_13_from_csv(csv_path, player_prefix="P1", gt_units="mm"):
    """
    Returns gt: (T,13,3) with joint order:
    [HEAD, RSHOULDER, LSHOULDER, RELBOW, LELBOW, LHIP, RHIP, RWRIST, LWRIST, RKNEE, LKNEE, RANKLE, LANKLE]
    Columns expected (like your file):
    P1:HEDO_x, P1:HEDO_y, P1:HEDO_depth, ...
    """
    df = pd.read_csv(csv_path)

    cols = []
    for code in GT_CODES:
        cols += [f"{player_prefix}:{code}_x", f"{player_prefix}:{code}_y", f"{player_prefix}:{code}_depth"]

    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing GT columns (showing first 10): {missing[:10]}")

    data = df[cols].to_numpy(dtype=np.float32)      
    gt = data.reshape(-1, 13, 3)                    

    if gt_units == "mm":
        gt = gt / 1000.0                            
    elif gt_units == "m":
        pass
    else:
        raise ValueError("gt_units must be 'mm' or 'm'")

    return gt

def root_center_13(skel13):
    """Root = mid-hip from RHIP(5) and LHIP(6) in your 13 order."""
    root = (skel13[:, 5, :] + skel13[:, 6, :]) / 2.0
    return skel13 - root[:, None, :]

def yaw_rotate_z(skel, yaw_deg):
    if abs(yaw_deg) < 1e-9:
        return skel
    theta = np.radians(yaw_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32)
    return skel @ R.T

def match_fps(pred, gt, fps_pred, fps_gt):
    if fps_pred == fps_gt:
        return pred, gt

    if fps_gt > fps_pred:
        ratio = fps_gt / fps_pred
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("Non-integer FPS ratio; use interpolation.")
        gt = gt[::int(round(ratio))]
    else:
        ratio = fps_pred / fps_gt
        if abs(ratio - round(ratio)) > 1e-6:
            raise ValueError("Non-integer FPS ratio; use interpolation.")
        pred = pred[::int(round(ratio))]

    return pred, gt

def plot_skeleton(pred_skel, gt_skel, frame_idx=50):
    """
    Visualize 3D skeletons for a specific frame.
    pred_skel, gt_skel: shape (N_frames, 13, 3)
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Select frame
    pred_frame = pred_skel[frame_idx] * 1000  # convert to mm
    gt_frame = gt_skel[frame_idx] * 1000
    
    # Define skeleton edges (connections between joints)
    skeleton = [
        (0, 1),  # HEAD ↔ RSHOULDER
        (1, 3),  # RSHOULDER ↔ RELBOW
        (3, 7),  # RELBOW ↔ RWRIST
        (0, 2),  # HEAD ↔ LSHOULDER
        (2, 4),  # LSHOULDER ↔ LELBOW
        (4, 8),  # LELBOW ↔ LWRIST
        (1, 2),  # RSHOULDER ↔ LSHOULDER
        (5, 6),  # RHIP ↔ LHIP
        (6, 9),  # RHIP ↔ RKNEE
        (5, 10), # LHIP ↔ LKNEE
        (9, 11), # RKNEE ↔ RANKLE
        (10, 12),# LKNEE ↔ LANKLE
        (6, 1),  # RHIP ↔ RSHOULDER
        (5, 2),  # LHIP ↔ LSHOULDER
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

    # Camera settings
    ax.view_init(elev=20, azim=-60)
    plt.tight_layout()
    plt.show()

def main(pred_path, gt_csv, fps_pred=50, fps_gt=50, yaw_deg=0.0, gt_units="mm", player="P1", plot=False):
    pred = np.load(pred_path, allow_pickle=True)["predictions"]  # (T,13,3) OR something else

    # Ensure (T,13,3)
    if pred.ndim == 2 and pred.shape[1] == 3:
        pred = pred.reshape(-1, 13, 3)
    elif pred.ndim != 3 or pred.shape[1:] != (13, 3):
        raise ValueError(f"Expected pred shape (T,13,3). Got {pred.shape}")

    gt = load_gt_13_from_csv(gt_csv, player_prefix=player, gt_units=gt_units)

    # FPS match
    pred, gt = match_fps(pred, gt, fps_pred, fps_gt)

    # Trim to same length (from the end, like your old script)
    T = min(len(pred), len(gt))
    pred = pred[-T:]
    gt   = gt[-T:]

    # Root-center both (mid-hip)
    pred_rc = root_center_13(pred)
    gt_rc   = root_center_13(gt)
    
    print(pred_rc.shape, gt_rc.shape)
    print(pred_rc[50], gt_rc[50])

    # Optional yaw rotation (to fix global frame mismatch)
    # pred_rc = yaw_rotate_z(pred_rc, yaw_deg)

    # MPJPE (meters->mm)
    errors = np.linalg.norm(pred_rc - gt_rc, axis=2)  # (T,13) in meters
    mpjpe_mm = float(np.mean(errors) * 1000.0)
    print(f"MPJPE: {mpjpe_mm:.2f} mm")

    # PCK@150mm
    thr_mm = 150.0
    pck = 100.0 * np.mean((errors * 1000.0) < thr_mm)
    print(f"PCK @ {thr_mm:.0f}mm: {pck:.2f}%")
    
    if plot:
        # Plot GT Skeleton
        plot_skeleton(pred_rc, gt_rc, frame_idx=50)

    # MPJPE over time
    if plot:
        mpjpe_t = np.mean(errors, axis=1) * 1000.0
        plt.figure(figsize=(9, 4))
        plt.plot(mpjpe_t)
        plt.title("MPJPE over time (mm)")
        plt.xlabel("Frame")
        plt.ylabel("MPJPE (mm)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", required=True, help="Path to predictions_3d_13.npz")
    ap.add_argument("--gt_csv", required=True, help="Path to GT csv (e.g. tennis_s1_joint_projections_and_depths.csv)")
    ap.add_argument("--fps_pred", type=int, default=50)
    ap.add_argument("--fps_gt", type=int, default=50)
    ap.add_argument("--yaw_deg", type=float, default=0.0, help="Optional yaw rotation around Z for prediction alignment.")
    ap.add_argument("--gt_units", choices=["mm","m"], default="mm")
    ap.add_argument("--player", default="P1", help="Player prefix in CSV (P1, P2, ...)")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    main(
        pred_path=args.pred_path,
        gt_csv=args.gt_csv,
        fps_pred=args.fps_pred,
        fps_gt=args.fps_gt,
        yaw_deg=args.yaw_deg,
        gt_units=args.gt_units,
        player=args.player,
        plot=args.plot,
    )