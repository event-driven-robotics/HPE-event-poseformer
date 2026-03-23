import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add common modules to path
sys.path.append(os.getcwd())

from common.model_poseformer import PoseTransformerV2
from common.camera import world_to_camera, normalize_screen_coordinates

def mpjpe(predicted, target):
    """
    Mean Per-Joint Position Error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many benchmarks.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def show_3d_pose(predicted, target, frame_idx):
    """
    Visualize the 3D pose for a specific frame.
    Input shapes: (J, 3)
    """
    fig = plt.figure(figsize=(10, 10))
    
    # Overlay Plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot GT first (Blue)
    _plot_single_pose(ax, target, 'blue', 'Ground Truth')
    
    # Plot Prediction (Red)
    _plot_single_pose(ax, predicted, 'red', 'Predicted')
    
    ax.set_title(f'Frame {frame_idx} Overlay\n(Blue=GT, Red=Pred)')
    ax.legend()
    
    # Set limits based on both
    all_pts = np.concatenate([predicted, target], axis=0)
    center = np.mean(all_pts, axis=0)
    radius = 10.0 # Standardized scale
    
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    plt.show()

def _plot_single_pose(ax, pose, color, label):
    # DHP19/H36M skeleton connections (approximate standard limb connections)
    # Joints: 0:Pelvis, 1:R_Hip, 2:R_Knee, 3:R_Ankle, 4:L_Hip, 5:L_Knee, 6:L_Ankle, 
    #         7:Torso, 8:Neck, 9:Nose, 10:Head, 11:L_Shoulder, 12:L_Elbow, 13:L_Wrist,
    #         14:R_Shoulder, 15:R_Elbow, 16:R_Wrist
    if pose.shape[0] == 13:
        # DHP19 13-joint skeleton (Verified from metadata)
        # Joints: ['hipL', 'head', 'shoulderR', 'shoulderL', 'elbowR', 'elbowL', 'hipR', 'handR', 'handL', 'kneeR', 'kneeL', 'footR', 'footL']
        # Indices: 0:hipL, 1:head, 2:shoR, 3:shoL, 4:elbR, 5:elbL, 6:hipR, 7:handR, 8:handL, 9:kneeR, 10:kneeL, 11:footR, 12:footL
        connections = [
            (0, 10), (10, 12),    # Left Leg: hipL -> kneeL -> footL
            (6, 9), (9, 11),      # Right Leg: hipR -> kneeR -> footR
            (3, 5), (5, 8),       # Left Arm: shoL -> elbL -> handL
            (2, 4), (4, 7),       # Right Arm: shoR -> elbR -> handR
            (0, 6),               # Hips: hipL -> hipR
            (3, 2),               # Shoulders: shoL -> shoR
            (0, 3), (6, 2),       # Torso: hipL->shoL, hipR->shoR (Box torso)
            (3, 1), (2, 1)        # Head: shoL->head, shoR->head
        ]
    elif pose.shape[0] == 17:
        # H36M 17-joint skeleton
        connections = [
            (0, 1), (1, 2), (2, 3),       # Right leg
            (0, 4), (4, 5), (5, 6),       # Left leg
            (0, 7), (7, 8), (8, 9), (9, 10), # Spine
            (8, 11), (11, 12), (12, 13),  # Left arm
            (8, 14), (14, 15), (15, 16)   # Right arm
        ]
    else:
        # Fallback: connect sequential just to show something
        connections = [(i, i+1) for i in range(pose.shape[0]-1)]

    for i, j in connections:
        ax.plot([pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]], color=color)

    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, label=label)
    
    # Set labels and aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Try to set equal aspect ratio roughly
    # DHP19 normalized scale is approx [-10, 10]
    radius = 10.0  
    ax.set_xlim3d([-radius, radius])
    ax.set_ylim3d([-radius, radius])
    ax.set_zlim3d([-radius, radius])

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    print("Loading data...")
    path_2d = os.path.join('PoseFormerV2-main/data', 'data_2d_dhp19_gt.npz')
    path_3d = os.path.join('PoseFormerV2-main/data', 'data_3d_dhp19.npz')

    try:
        data_2d = np.load(path_2d, allow_pickle=True)['positions_2d'].item()
        data_3d = np.load(path_3d, allow_pickle=True)['positions_3d'].item()
    except FileNotFoundError:
        print("Error: DHP19 data files not found in 'DHP19/new' directory.")
        return

    # Select a sample: Subject S13, Action 'Standard Walk', Camera 2 (idx 1 usually)
    # Note: Actions in key files often have specific names like 'Standard Walk 1'
    subject = 'S14'
    action_prefix = '4_1' 
    
    # Find exact action name
    action = None
    if subject in data_3d:
        for k in data_3d[subject].keys():
            if k.startswith(action_prefix):
                action = k
                break
    
    if action is None:
        print(f"Could not find action starting with '{action_prefix}' for subject '{subject}'.")
        # Fallback to first available
        subject = list(data_3d.keys())[0]
        action = list(data_3d[subject].keys())[0]
        print(f"Falling back to: {subject}, {action}")

    camera_idx = 1 # Camera 2
    
    # Extract sequences
    # 2D Input: (Frames, 17, 2) or (Frames, 13, 2)
    # 3D GT: (Frames, 17, 3) or (Frames, 13, 3)
    kps_2d = data_2d[subject][action][camera_idx]
    pos_3d = data_3d[subject][action] # 3D is global per action
    
    print(f"Original 2D shape: {kps_2d.shape}")
    print(f"Original 3D shape: {pos_3d.shape}")

    # Ensure 3D shape (Frames, Joints, Coords)
    if kps_2d.ndim == 2:
        if kps_2d.shape[1] == 2: # (Joints, 2)
            kps_2d = kps_2d[np.newaxis, :, :]
    
    if pos_3d.ndim == 2:
        pos_3d = pos_3d[np.newaxis, :, :]
        
    print(f"Reshaped 2D shape: {kps_2d.shape}")
    print(f"Reshaped 3D shape: {pos_3d.shape}")
    
    # Normalize 3D GT (DHP19 is in mm, usually need /100 or /1000 depending on training)
    # Training script divides by 100 for DHP19 to normalize to ~meter-ish scale
    pos_3d_norm = pos_3d.copy() / 100.0

    # Normalize 2D Input (Screen coordinates -> [-1, 1])
    # DHP19 resolution: 346x260
    kps_2d_norm = normalize_screen_coordinates(kps_2d[..., :2], w=346, h=260)
    
    print(f"Loaded sequence. Frames: {kps_2d.shape[0]}")

    # 2. Setup Model
    print("Loading model...")
    # These args must match training config
    class Config:
        number_of_frames = 243
        number_of_kept_frames = 27
        number_of_kept_coeffs = 27
        number_of_joints = 13
        dropout = 0.0
        depth = 4
        embed_dim_ratio = 32
        
    config = Config()
    
    model = PoseTransformerV2(
        num_frame=config.number_of_frames,
        num_joints=config.number_of_joints,
        in_chans=2,
        num_heads=8,
        mlp_ratio=2.,
        qkv_bias=True,
        qk_scale=None,
        drop_path_rate=0.,
        args=config # Pass config as args
    ).to(device)

    # Load Checkpoint
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at: {checkpoint_path}")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle state dict (sometimes wrapped in 'model_pos', sometimes has 'module.' prefix)
    state_dict = checkpoint.get('model_pos', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove 'module.'
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    # 3. Inference
    print("Running inference...")
    
    # Prepare input batch
    # Needs to be (Batch, Frames, Joints, 2)
    # We will do a sliding window or just take the first N frames
    
    receptive_field = config.number_of_frames
    pad = (receptive_field - 1) // 2
    
    # Pad the sequence for receptive field
    # Simple edge padding
    input_2d = np.pad(kps_2d_norm, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    
    predicted_3d = []
    
    # Process batch-wise to be safe, or just one big batch if it fits
    # Let's process valid frames
    num_frames = kps_2d_norm.shape[0]
    
    # Create windows
    # Shape: (Num_Frames, Receptive_Field, 17, 2)
    inputs_tensor = []
    for i in range(num_frames):
        window = input_2d[i : i + receptive_field]
        inputs_tensor.append(window)
        
    inputs_tensor = np.array(inputs_tensor)
    inputs_tensor = torch.from_numpy(inputs_tensor).float().to(device)
    
    # Run in batches to avoid OOM
    batch_size = 512
    with torch.no_grad():
        for i in range(0, num_frames, batch_size):
            batch = inputs_tensor[i : i + batch_size]
            output = model(batch)
            predicted_3d.append(output.cpu().numpy())
            
    predicted_3d = np.concatenate(predicted_3d, axis=0) # (Frames, 1, 17, 3) (center frame)
    predicted_3d = predicted_3d[:, 0, :, :] # (Frames, 17, 3)

    # Center at root
    # Protocol #1: MPJPE (Mean Per Joint Position Error)
    # Align root joint.
    
    if predicted_3d.shape[1] == 13:
        pred_centered = predicted_3d - predicted_3d[:, :1, :]
        gt_centered = pos_3d_norm - pos_3d_norm[:, :1, :]
        # # DHP19: 0=HipL, 6=HipR
        # # Calculate Pelvis for Pred
        # root_pred = (predicted_3d[:, 0, :] + predicted_3d[:, 6, :]) / 2.0
        # root_pred = root_pred[:, np.newaxis, :] # (Frames, 1, 3)
        
        # # Calculate Pelvis for GT
        # root_gt = (pos_3d_norm[:, 0, :] + pos_3d_norm[:, 6, :]) / 2.0
        # root_gt = root_gt[:, np.newaxis, :]
        
        # pred_centered = predicted_3d - root_pred
        # gt_centered = pos_3d_norm - root_gt
    else:
        # Default/H36M: Root is usually idx 0
        pred_centered = predicted_3d - predicted_3d[:, :1, :]
        gt_centered = pos_3d_norm - pos_3d_norm[:, :1, :]
    
    print(f"\nStats - Pred Centered: Min {np.min(pred_centered):.2f}, Max {np.max(pred_centered):.2f}, Mean {np.mean(pred_centered):.2f}")
    print(f"Stats - GT Centered: Min {np.min(gt_centered):.2f}, Max {np.max(gt_centered):.2f}, Mean {np.mean(gt_centered):.2f}")

    # Handle Frame Mismatch (e.g. Single Frame GT vs Sequence Pred)
    num_eval = min(pred_centered.shape[0], gt_centered.shape[0])
    if num_eval < pred_centered.shape[0]:
        print(f"Warning: Evaluating on first {num_eval} frames only (GT frames: {gt_centered.shape[0]})")
        
    pred_eval = pred_centered[:num_eval]
    gt_eval = gt_centered[:num_eval]
    
    # Convert back to mm for error reporting
    # DHP19 scale factor used in training was /100, so *100 gives back mm
    error_mm = np.mean(np.linalg.norm(pred_eval - gt_eval, axis=2)) * 100.0
    
    print(f"\nEvaluation Results:")
    print(f"MPJPE (Protocol #1): {error_mm:.2f} mm")
 
    # 5. Visualization
    if args.plot:
        print("\nPlotting result for frame 100...")
        frame_to_show = min(100, num_frames - 1)
        show_3d_pose(pred_centered[frame_to_show], gt_centered[frame_to_show], frame_to_show)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--plot', action='store_true', default=True, help='Show 3D plot')
    
    args = parser.parse_args()
    main(args)
