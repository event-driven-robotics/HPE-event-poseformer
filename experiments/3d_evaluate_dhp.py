import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

def p_mpjpe(predicted, target):
    """
    Mean Per Joint Position Error after rigid alignment (scale, rotation, and translation).
    """
    muX = np.mean(target, axis=1, keepdims=True)
    muY = np.mean(predicted, axis=1, keepdims=True)
    
    X0 = target - muX
    Y0 = predicted - muY

    normX = np.sqrt(np.sum(X0**2, axis=(1, 2), keepdims=True))
    normY = np.sqrt(np.sum(Y0**2, axis=(1, 2), keepdims=True))
    
    X0 /= normX
    Y0 /= normY

    H = np.matmul(X0.transpose(0, 2, 1), Y0)
    
    if not np.all(np.isfinite(H)):
        return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))

    try:
        U, s, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))

    V = Vt.transpose(0, 2, 1)
    R = np.matmul(V, U.transpose(0, 2, 1))

    sign_detR = np.sign(np.expand_dims(np.linalg.det(R), axis=1))
    V[:, :, -1] *= sign_detR
    s[:, -1] *= sign_detR.flatten()
    R = np.matmul(V, U.transpose(0, 2, 1))

    tr = np.expand_dims(np.sum(s, axis=1, keepdims=True), axis=2)
    a = tr * normX / normY
    t = muX - a*np.matmul(muY, R)
    
    predicted_aligned = a*np.matmul(predicted, R) + t
    return predicted_aligned, np.mean(np.linalg.norm(predicted_aligned - target, axis=2))

def main(pred_path, subject_name, activity_name, frame_idx=0, plot=False, skip_pred=0, skip_gt=0, max_eval_frames=None, swap_yz=False):
    # === Load predictions ===
    predictions = np.load(pred_path, allow_pickle=True)['predictions']

    # === Load ground truth data ===
    gt_path = "DHP19/fixed_gt/data_3d_dhp19.npz"
    if not os.path.exists(gt_path):
        gt_path = "PoseFormerV2-main/data/data_3d_dhp19.npz"
        
    print(f"Loading GT from {gt_path}...")
    data = np.load(gt_path, allow_pickle=True)
    positions_3d = data["positions_3d"].item()

    if subject_name not in positions_3d:
        raise KeyError(f"Subject {subject_name} not found in GT. Available subjects: {list(positions_3d.keys())}")
    subject = positions_3d[subject_name]
    
    if activity_name not in subject:
        raise KeyError(f"Activity {activity_name} not found for {subject_name}. Available: {list(subject.keys())}")
        
    gt_poses = subject[activity_name]
    if isinstance(gt_poses, dict) and 'positions_3d' in gt_poses:
        gt_poses = gt_poses['positions_3d'][0]
    
    print(f"GT original shape: {gt_poses.shape}")

    # === Joint reordering/reshape ===
    if predictions.ndim == 2 and predictions.shape[1] == 3:
        if predictions.shape[0] % 13 == 0:
            num_frames = predictions.shape[0] // 13
            predictions = predictions.reshape(num_frames, 13, 3)
        elif predictions.shape[0] % 17 == 0:
            num_frames = predictions.shape[0] // 17
            print(f"WARNING: Predictions have 17 joints. Reshaping to (N, 17, 3). MPJPE will fail if GT has 13.")
            predictions = predictions.reshape(num_frames, 17, 3)
    
    # Auto-scale check
    if predictions.max() < 20: 
        print("INFO: Predictions seem to be in normalized scale [0-16]. Scaling by 100 to mm.")
        predictions *= 100
    elif predictions.max() < 2:
        print("INFO: Predictions seem to be in meters. Scaling by 1000 to mm.")
        predictions *= 1000

    if swap_yz:
        print("INFO: Swapping Y and Z axes for predictions.")
        tmp = predictions[:, :, 1].copy()
        predictions[:, :, 1] = predictions[:, :, 2]
        predictions[:, :, 2] = tmp

    # === Frame synchronization ===
    # GT in data_3d_dhp19.npz is already at 50fps (matching MoveEnet output rate)
    print(f"GT shape: {gt_poses.shape}")

    # Apply skips
    if skip_pred > 0:
        predictions = predictions[skip_pred:]
    if skip_gt > 0:
        gt_poses = gt_poses[skip_gt:]
        
    num_frames = min(predictions.shape[0], gt_poses.shape[0])
    if max_eval_frames is not None:
        num_frames = min(num_frames, max_eval_frames)
        
    print(f"Sync: Evaluating {num_frames} frames (Skip Pred: {skip_pred}, Skip GT: {skip_gt})")
    
    pred_xyz = predictions[:num_frames]
    gt_xyz = gt_poses[:num_frames]

    # Root-relative alignment for MPJPE
    pred_root_rel = pred_xyz - pred_xyz[:, [0], :]
    gt_root_rel = gt_xyz - gt_xyz[:, [0], :]

    # Protocol 1: MPJPE (Root-relative only)
    errors = np.linalg.norm(pred_root_rel - gt_root_rel, axis=2)
    mpjpe = np.mean(errors)
    
    # Protocol 2: P-MPJPE (Rigid alignment)
    p_pred_aligned, p_mpjpe_val = p_mpjpe(pred_xyz, gt_xyz)

    print(f"\n--- Results (mm) ---")
    print(f"MPJPE:   {mpjpe:.2f} mm")
    print(f"P-MPJPE: {p_mpjpe_val:.2f} mm")
    
    # PCK@150mm
    threshold = 150
    correct_kps = errors < threshold
    pck = 100.0 * np.sum(correct_kps) / correct_kps.size
    print(f"PCK@150: {pck:.2f}%")
    
    # Print ranges for debugging
    print(f"Pred Range: X [{pred_xyz[..., 0].min():.1f}, {pred_xyz[..., 0].max():.1f}], "
          f"Y [{pred_xyz[..., 1].min():.1f}, {pred_xyz[..., 1].max():.1f}], "
          f"Z [{pred_xyz[..., 2].min():.1f}, {pred_xyz[..., 2].max():.1f}]")
    print(f"GT Range:   X [{gt_xyz[..., 0].min():.1f}, {gt_xyz[..., 0].max():.1f}], "
          f"Y [{gt_xyz[..., 1].min():.1f}, {gt_xyz[..., 1].max():.1f}], "
          f"Z [{gt_xyz[..., 2].min():.1f}, {gt_xyz[..., 2].max():.1f}]")

    # Height Check
    determine_height(gt_root_rel, "Ground Truth", frame_idx % num_frames)
    determine_height(pred_root_rel, "Prediction", frame_idx % num_frames)
    
    if plot:
        mpjpe_values = np.mean(np.linalg.norm(pred_root_rel - gt_root_rel, axis=2), axis=1)
        plt.figure(figsize=(10, 5))
        plt.plot(mpjpe_values, label='MPJPE')
        plt.title(f"MPJPE over time ({subject_name} {activity_name})")
        plt.xlabel("Frame")
        plt.ylabel("mm")
        plt.grid(True)
        plt.show()

    # Visualization
    plot_skeleton_3d_comparison(pred_root_rel, gt_root_rel, frame_idx % num_frames, p_pred_aligned - p_pred_aligned[:, [0], :])

def determine_height(skel, label, f_idx):
    head_idx = 1
    ank_r = 11
    ank_l = 12
    foot_z = (skel[f_idx, ank_r, 2] + skel[f_idx, ank_l, 2]) / 2
    head_z = skel[f_idx, head_idx, 2]
    h = abs(head_z - foot_z)
    print(f"Height ({label}): {h:.1f} mm")

def plot_skeleton_3d_comparison(pred, gt, f_idx, prot2_pred=None):
    fig = plt.figure(figsize=(15, 7))
    
    # Subplot 1: Root-relative (Direct comparison)
    ax1 = fig.add_subplot(121, projection='3d')
    draw_skel(ax1, gt[f_idx], 'blue', 'GT', alpha=0.5)
    draw_skel(ax1, pred[f_idx], 'red', 'Pred')
    ax1.set_title("Root-Relative Comparison (P1)")
    
    # Subplot 2: Rigidly aligned (P2)
    if prot2_pred is not None:
        ax2 = fig.add_subplot(122, projection='3d')
        draw_skel(ax2, gt[f_idx], 'blue', 'GT', alpha=0.5)
        draw_skel(ax2, prot2_pred[f_idx], 'green', 'Aligned Pred')
        ax2.set_title("Rigidly Aligned Comparison (P2)")
    
    plt.tight_layout()
    plt.show()

def draw_skel(ax, joints, color, label, alpha=1.0):
    I = np.array([0, 10, 6, 9, 3, 5, 2, 4, 0, 3, 0, 6, 3, 2])
    J = np.array([10, 12, 9, 11, 5, 8, 4, 7, 6, 2, 3, 2, 1, 1])
    for i, j in zip(I, J):
        ax.plot([joints[i,0], joints[j,0]], [joints[i,1], joints[j,1]], [joints[i,2], joints[j,2]], 
                color=color, alpha=alpha, linewidth=2)
    ax.scatter(joints[:,0], joints[:,1], joints[:,2], color=color, alpha=alpha, s=20)
    
    # Aspect ratio
    r = 1000 # 1 meter radius
    ax.set_xlim3d([-r, r])
    ax.set_ylim3d([-r, r])
    ax.set_zlim3d([-r, r])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_path", type=str, required=True)
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--activity", type=str, required=True)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--plot", action='store_true')
    parser.add_argument("--skip_pred", type=int, default=0)
    parser.add_argument("--skip_gt", type=int, default=0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--swap_yz", action='store_true')

    args = parser.parse_args()
    main(args.pred_path, args.subject, args.activity, args.frame, args.plot, 
         args.skip_pred, args.skip_gt, args.max_frames, args.swap_yz)
