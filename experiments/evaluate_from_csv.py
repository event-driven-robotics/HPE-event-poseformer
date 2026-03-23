import numpy as np
import torch
import torch.nn as nn
import os
import sys
import argparse
# import pandas as pd # Removed due to env issues
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path to allow imports from common
sys.path.append(os.path.join(os.getcwd(), 'PoseFormerV2-main'))

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates
from common.model_poseformer import PoseTransformerV2
from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error
from common.generators import UnchunkedGenerator
from common.dhp19_dataset import Dhp19Dataset

def parse_csv_path(csv_path):
    """
    Extract subject, action, and camera from path like:
    outputs/dhp19/S13_1_1/moveEnet_keypoints.csv
    """
    pattern = r'S(\d+)_(\d+)_(\d+)'
    match = re.search(pattern, csv_path)
    if match:
        subject = f"S{match.group(1)}"
        action_idx = match.group(2)
        camera_idx = int(match.group(3)) - 1 # 1-based to 0-based
        return subject, action_idx, camera_idx
    return None, None, None

def load_csv_keypoints(csv_path, mapping=None):
    # Expected columns: frame, joint, x, y
    # Skipping header (1 line)
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    # data columns: 0:frame, 1:joint, 2:x, 3:y
    frames = data[:, 0].astype(int)
    joints = data[:, 1].astype(int)
    
    num_frames = np.max(frames) + 1
    num_joints = np.max(joints) + 1
    
    # We always expect 13 joints for DHP19 model
    actual_num_joints = 13
    keypoints = np.zeros((num_frames, actual_num_joints, 2))
    for i in range(len(data)):
        f = int(data[i, 0])
        j = int(data[i, 1])
        if mapping is not None:
            if j in mapping:
                target_j = mapping[j]
                keypoints[f, target_j, 0] = data[i, 2]
                keypoints[f, target_j, 1] = data[i, 3]
        elif j < actual_num_joints:
            keypoints[f, j, 0] = data[i, 2]
            keypoints[f, j, 1] = data[i, 3]
        
    return keypoints

def show_3d_pose(predicted, target, frame_idx, save_path='evaluate_viz.png'):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # DHP19 connections
    connections = [
        (0, 10), (10, 12), (6, 9), (9, 11), (3, 5), (5, 8), (2, 4), (4, 7),
        (0, 6), (3, 2), (0, 3), (6, 2), (3, 1), (2, 1)
    ]

    def plot_pose(pose, color, label):
        for i, j in connections:
            ax.plot([pose[i, 0], pose[j, 0]],
                    [pose[i, 1], pose[j, 1]],
                    [pose[i, 2], pose[j, 2]], color=color)
        ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, label=label)

    plot_pose(target, 'blue', 'Ground Truth')
    plot_pose(predicted, 'red', 'Predicted')
    
    ax.set_title(f'Frame {frame_idx} Comparison\n(Blue=GT, Red=Pred)')
    ax.legend()
    
    all_pts = np.concatenate([predicted, target], axis=0)
    center = np.mean(all_pts, axis=0)
    radius = np.max(np.linalg.norm(all_pts - center, axis=1)) * 1.1
    
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    ax.set_box_aspect([1, 1, 1])
    
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Evaluate PoseFormerV2 from CSV detections')
    parser.add_argument('--csv', type=str, required=True, help='Path to keypoints CSV')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--camera', type=int, default=None, help='Manual camera index (0-3)')
    parser.add_argument('--num-frames', type=int, default=27, help='Receptive field')
    parser.add_argument('--viz-frame', type=int, default=100, help='Frame to visualize')
    parser.add_argument('--no-flip-y', action='store_true', help='Disable Y-axis flip (260 - y)')
    parser.add_argument('--debug', action='store_true', help='Print debug info')
    
    # Avoid sys.argv conflicts with parse_args()
    # We'll pass an empty list or only recognized global args to parse_args()
    # But since we need parse_args() to return the default object for model construction:
    orig_argv = sys.argv
    sys.argv = [orig_argv[0]] # Keep only script name for parse_args() defaults
    args = parse_args()
    sys.argv = orig_argv # Restore
    
    cmd_args = parser.parse_args()
    args.number_of_frames = cmd_args.num_frames
    args.number_of_frames = cmd_args.num_frames
    
    # 1. Parse CSV Metadata
    subject, action_idx, camera_idx = parse_csv_path(cmd_args.csv)
    if cmd_args.camera is not None:
        camera_idx = cmd_args.camera
        
    if subject is None or action_idx is None or camera_idx is None:
        print("Error: Could not determine sequence metadata from path. Use --camera manually.")
        return

    print(f"Evaluating: Subject={subject}, ActionIdx={action_idx}, CameraIdx={camera_idx}")

    # 2. Load 3D Dataset
    dataset_path = 'DHP19/fixed_gt/data_3d_dhp19.npz'
    if not os.path.exists(dataset_path):
        dataset_path = 'PoseFormerV2-main/data/data_3d_dhp19.npz'
    
    print(f"Loading 3D GT from {dataset_path}...")
    dataset = Dhp19Dataset(dataset_path)
    
    # Find the full action name in dataset
    target_action = None
    for action in dataset[subject].keys():
        if action.startswith(action_idx):
            target_action = action
            break
            
    if target_action is None:
        print(f"Error: Action index {action_idx} not found for {subject}")
        return

    # Load and normalize 3D GT (matching evaluate_model.py logic)
    poses_3d = dataset[subject][target_action]['positions_3d'][camera_idx]
    # Scaled
    poses_3d = poses_3d / 100.0

    # 3. Load and Normalize 2D CSV
    # Mapping MoveNet-13 to DHP19 ORDER_13
    # MoveNet-13 order often seen in this project: [nose, sL, sR, eL, eR, wL, wR, hL, hR, kL, kR, aL, aR]
    # DHP19 order: [hipL, head, shoulderR, shoulderL, elbowR, elbowL, hipR, wristR, wristL, kneeR, kneeL, ankleR, ankleL]
    # Standard MoveNet order typically: [nose, sL, sR, eL, eR, wL, wR, hL, hR, kL, kR, aL, aR]
    # DHP19 order: [hipL, head, shoulderR, shoulderL, elbowR, elbowL, hipR, wristR, wristL, kneeR, kneeL, ankleR, ankleL]
    # verified mapping based on visualize_csv.py and GT inspection
    # CSV order: 0:Head, 1:ShL, 2:ShR, 3:ElL, 4:ElR, 5:HiL, 6:HiR, 7:WrL, 8:WrR, 9:KnL, 10:KnR, 11:AnL, 12:AnR
    # DHP19 L/R is person-centric (generally swapped in image for these cameras)
    movenet_to_dhp19 = {
        0: 1,   # Head -> Head
        1: 2,   # ShL -> ShR
        2: 3,   # ShR -> ShL
        3: 4,   # ElL -> ElR
        4: 5,   # ElR -> ElL
        5: 6,   # HiL -> HiR
        6: 0,   # HiR -> HiL (ROOT)
        7: 7,   # WrL -> WrR
        8: 8,   # WrR -> WrL
        9: 9,   # KnL -> KnR
        10: 10, # KnR -> KnL
        11: 11, # AnL -> AnR
        12: 12  # AnR -> AnL
    }
    
    print(f"Loading 2D keypoints from {cmd_args.csv} with reordering...")
    poses_2d = load_csv_keypoints(cmd_args.csv, mapping=movenet_to_dhp19)
    
    # Ensure sync
    if len(poses_2d) != len(poses_3d):
        print(f"Warning: Frame count mismatch. CSV: {len(poses_2d)}, GT: {len(poses_3d)}")
        min_len = min(len(poses_2d), len(poses_3d))
        poses_2d = poses_2d[:min_len]
        poses_3d = poses_3d[:min_len]

    # Normalize 2D
    # MoveEnet keypoints often need a Y-flip for DHP19 if they were extracted with standard origin
    # Previous tests showed 260 - y works best for alignment
    cam = dataset.cameras()[subject][camera_idx]
    poses_2d_norm = poses_2d.copy()
    # Restore Y-flip: CSV pixels are Y-down, but model expects Y-up (260 - y)
    print("Applying Y-axis flip (260 - y)...")
    poses_2d_norm[..., 1] = 260.0 - poses_2d_norm[..., 1]

    w, h = cam['res_w'], cam['res_h']
    print(f"Normalizing with resolution: {w}x{h}")
    poses_2d_norm[..., :2] = normalize_screen_coordinates(poses_2d_norm[..., :2], w=w, h=h)

    # Restore 2D centering: index 6 (MoveNet HiR) is DHP19 index 0 (ROOT)
    print("Centering 2D inputs at MoveNet index 6 (DHP19 root)...")
    poses_2d_norm = poses_2d_norm - poses_2d_norm[:, 6:7, :]

    # 4. Model Setup
    print(f"Loading model from {cmd_args.checkpoint}...")
    receptive_field = args.number_of_frames
    num_joints = poses_2d.shape[1]
    
    model = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
                              num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, 
                              drop_path_rate=0, args=args)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(model).to(device)
    
    try:
        checkpoint = torch.load(cmd_args.checkpoint, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older torch versions without weights_only
        checkpoint = torch.load(cmd_args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    model.eval()

    # 5. Run Inference
    pad = (receptive_field - 1) // 2
    # Use UnchunkedGenerator with single sequence
    gen = UnchunkedGenerator(None, [poses_3d], [poses_2d_norm], pad=pad)
    
    # Symmetry for DHP19 (Matching evaluate_model.py exactly)
    joints_left = [0, 3, 5, 8, 10, 12]
    joints_right = [6, 2, 4, 7, 9, 11]
    
    # 2D Symmetry for MoveNet-13 (Pairs: 1<->2, 3<->4, 5<->6, 7<->8, 9<->10, 11<->12)
    kps_left = [1, 3, 5, 7, 9, 11]
    kps_right = [2, 4, 6, 8, 10, 12]

    print("Running inference (with TTA inspiration from evaluate_model.py)...")
    predicted_all = []
    target_all = []
    
    with torch.no_grad():
        for _, batch_3d, batch_2d in gen.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32')).to(device)
            inputs_3d = torch.from_numpy(batch_3d.astype('float32')).to(device)
            
            # Prepare data (sliding window)
            inputs_2d_p = torch.squeeze(inputs_2d)
            if inputs_2d_p.dim() == 2: # Single frame edge case
                inputs_2d_p = inputs_2d_p.unsqueeze(0)
            
            out_num = inputs_2d_p.shape[0] - receptive_field + 1
            eval_input = torch.empty(out_num, receptive_field, num_joints, 2).to(device)
            for i in range(out_num):
                eval_input[i] = inputs_2d_p[i:i+receptive_field]
            
            # Test-time augmentation (Flip)
            eval_input_flip = eval_input.clone()
            eval_input_flip[..., 0] *= -1 # Flip X
            eval_input_flip[:, :, kps_left + kps_right, :] = eval_input_flip[:, :, kps_right + kps_left, :]
            
            # Forward Pass
            pred = model(eval_input)
            pred_flip = model(eval_input_flip)

            # Flip back
            pred_flip[..., 0] *= -1
            if pred_flip.dim() == 4:
                pred_flip[:, :, joints_left + joints_right] = pred_flip[:, :, joints_right + joints_left]
            else:
                pred_flip[:, joints_left + joints_right] = pred_flip[:, joints_right + joints_left]

            # Average (Consistency with evaluate_model.py averaging)
            pred = (pred + pred_flip) / 2.0
            
            # Raw output debug (before centering)
            if pred.dim() == 4:
                raw_head = pred[0, 0, 1].cpu().numpy()
            else:
                raw_head = pred[0, 1].cpu().numpy()
            print(f"DEBUG: Raw model output (first frame, head [1]):\n{raw_head * 100}")
            
            # Consistency with evaluate_model.py: Skeletons evaluated root-relative
            target = inputs_3d.squeeze(0)[:pred.shape[0]].clone()
            target -= target[:, 0:1, :] if target.dim() == 3 else target[:, :, 0:1, :]
            
            # Center root at index 0
            if pred.dim() == 4:
                pred -= pred[:, :, 0:1, :]
            else:
                pred -= pred[:, 0:1, :]
            
            # Centered output debug
            if pred.dim() == 4:
                centered_head = pred[0, 0, 1].cpu().numpy()
            else:
                centered_head = pred[0, 1].cpu().numpy()
            print(f"DEBUG: Root-centered model output (first frame, head [1]):\n{centered_head * 100}")
            
            predicted_all.append(pred.cpu().numpy())
            target_all.append(target.cpu().numpy())

    predicted_all = np.concatenate(predicted_all, axis=0) # (N, J, 3)
    target_all = np.concatenate(target_all, axis=0) # (N, J, 3)
    
    # Squeeze the middle dim from predicted if exists
    if predicted_all.ndim == 4:
        predicted_all = predicted_all.squeeze(1)
        
    # Scale back to mm (DHP19 uses 100.0)
    scale = 100.0

    if len(predicted_all) != len(target_all):
        print(f"Error: Final shape mismatch. Pred: {len(predicted_all)}, Target: {len(target_all)}")
        min_len = min(len(predicted_all), len(target_all))
        predicted_all = predicted_all[:min_len]
        target_all = target_all[:min_len]

    mpjpe_val = np.mean(np.linalg.norm((predicted_all - target_all) * scale, axis=-1))
    
    print(f"\nResults for {cmd_args.csv}:")
    print(f"MPJPE: {mpjpe_val:.2f} mm")
    
    # 7. Visualization
    if cmd_args.viz_frame < len(predicted_all):
        p_frame = predicted_all[cmd_args.viz_frame]
        t_frame = target_all[cmd_args.viz_frame]
        
        show_3d_pose(p_frame * scale, t_frame * scale, cmd_args.viz_frame)

if __name__ == "__main__":
    main()
