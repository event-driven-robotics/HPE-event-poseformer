import numpy as np
import torch
import torch.nn as nn
import os
import sys
import argparse
import logging
from tqdm import tqdm
from time import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Add current directory to path to allow imports from common
sys.path.append(os.path.join(os.getcwd(), 'PoseFormerV2-main'))

import matplotlib
print(f"DEBUG: Matplotlib backend: {matplotlib.get_backend()}")

from common.arguments import parse_args
from common.camera import normalize_screen_coordinates, camera_to_world, image_coordinates, world_to_camera
from common.model_poseformer import PoseTransformerV2
from common.loss import mpjpe, p_mpjpe, n_mpjpe, mean_velocity_error
from common.generators import UnchunkedGenerator
from common.dhp19_dataset import Dhp19Dataset
from common.utils import deterministic_random

def load_2d_predictions(csv_file, num_joints, mapping=None, frame_column='frame', joint_column='joint', x_column='x', y_column='y'):
    
    df = pd.read_csv(csv_file)
    
    frames = df[frame_column].unique()
    actual_num_joints = max(num_joints, max(mapping.values()) + 1) if mapping else num_joints
    predictions = np.zeros((len(frames), actual_num_joints, 2)) 

    for i, frame in enumerate(frames):
        frame_data = df[df[frame_column] == frame]
        for j in range(num_joints):
            joint_data = frame_data[frame_data[joint_column] == j]
            if joint_data.empty:
                continue
            
            target_j = mapping[j] if (mapping and j in mapping) else j
            if target_j < actual_num_joints:
                predictions[i, target_j, 0] = joint_data[x_column].values[0]  # x coordinate
                predictions[i, target_j, 1] = joint_data[y_column].values[0]  # y coordinate

    return predictions


def format_movenet_csv(input_path, output_path):
    
    df_split = pd.read_csv(input_path, header=None, delimiter=" ")

    columns = []
    for i in range(1, 14):
        columns.append(f'joint_{i}_x')
        columns.append(f'joint_{i}_y')

    df_split.columns = ['timestamp', 'delay'] + columns

    df_sorted = df_split.sort_values(by='timestamp').reset_index(drop=True)

    # Create a new dataframe to store Frame, Joint, x, y
    frame_data = []

    # Loop through the rows and extract frame number, joint index, x, and y values
    for frame_number, row in df_sorted.iterrows():
        for joint in range(13):  # 13 joints, indexing from 0
            # Extract x and y for the current joint
            joint_x = row[f'joint_{joint + 1}_x']  # The x value for the joint
            joint_y = row[f'joint_{joint + 1}_y']  # The y value for the joint

            # Append to frame_data with the correct format
            frame_data.append([frame_number, joint, joint_x, joint_y])

    # Convert the data into a new DataFrame
    df_final = pd.DataFrame(frame_data, columns=['frame', 'joint', 'x', 'y'])

    # Save the processed data to the output path
    df_final.to_csv(output_path, index=False)
    print(f"DEBUG: Formatted CSV saved to {output_path}")

def fetch(dataset, keypoints, args, subjects, action_filter=None, subset=1, parse_3d_poses=True, load_gt=False):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    
    for subject in subjects:
        if subject not in keypoints:
            print(f"DEBUG: Subject {subject} not in keypoints")
            continue
        keys = keypoints[subject].keys()
        for action in keys:
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]

            # If parsing 3D poses, we must ensure synchronization and validity
            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                
                for i in range(len(poses_2d)): # Iterate across cameras
                    if poses_2d[i] is None:
                        continue
                    
                    # Ensure frames match
                    p2d = poses_2d[i]
                    p3d = poses_3d[i]
                    if len(p2d) != len(p3d):
                        min_len = min(len(p2d), len(p3d))
                        p2d = p2d[:min_len]
                        p3d = p3d[:min_len]

                    # Check for NaNs
                    valid_3d = not np.isnan(p3d).any()
                    valid_2d = not np.isnan(p2d).any()
                    
                    if not valid_3d or not valid_2d:
                        continue
                        
                    out_poses_3d.append(p3d)
                    out_poses_2d.append(p2d)
                    
                    if subject in dataset.cameras():
                         cams = dataset.cameras()[subject]
                         if i < len(cams) and 'intrinsic' in cams[i]:
                             out_camera_params.append(cams[i]['intrinsic'])
            else:
                for i in range(len(poses_2d)):
                     if poses_2d[i] is None or np.isnan(poses_2d[i]).any():
                         continue 
                     out_poses_2d.append(poses_2d[i])
                     if subject in dataset.cameras():
                         cams = dataset.cameras()[subject]
                         if i < len(cams) and 'intrinsic' in cams[i]:
                             out_camera_params.append(cams[i]['intrinsic'])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    return out_camera_params, out_poses_3d, out_poses_2d

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p

# Protocol 2: Rigid Alignment (Procrustes)
def p_mpjpe_with_coords(predicted, target):
    """
    Mean Per Joint Position Error after rigid alignment (scale, rotation, and translation).
    Returns (aligned_predicted, mpjpe).
    """
    assert predicted.shape == target.shape
    
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
        return predicted, np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))

    try:
        U, s, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return predicted, np.mean(np.linalg.norm(predicted - target, axis=len(target.shape)-1))

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
    # set the 0th joint of each frame in predicted_aligned to 0 - other joints stay the same
    predicted_aligned[:, 0, :] = 0
    return predicted_aligned, np.mean(np.linalg.norm(predicted_aligned - target, axis=len(target.shape)-1))

def show_3d_pose(predicted, target, frame_idx):
    """
    Visualize the 3D pose for a specific frame.
    Input shapes: (J, 3)
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot GT (Blue)
    _plot_single_pose(ax, target, 'blue', 'Ground Truth')
    
    # Plot Prediction (Red)
    _plot_single_pose(ax, predicted, 'red', 'Prediction')
    
    ax.set_title(f'Frame {frame_idx} Overlay\n(Blue=GT, Red=Prediction)')
    ax.legend()
    
    # Set limits
    all_pts = np.concatenate([predicted, target], axis=0)
    center = np.mean(all_pts, axis=0)
    radius = np.max(np.linalg.norm(all_pts - center, axis=1)) * 1.5 
    
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    # Aspect ratio should be equal
    ax.set_box_aspect([1, 1, 1])
    
    plt.show()

def _plot_single_pose(ax, pose, color, label):
    if pose.shape[0] == 13:
        # DHP19 joints
        # 0:hipL, 1:head, 2:shR, 3:shL, 4:elR, 5:elL, 6:hipR, 7:wrR, 8:wrL, 9:knR, 10:knL, 11:anR, 12:anL
        connections = [
            (0, 10), (10, 12),              # HipL -> KneeL -> AnkleL
            (6, 9), (9, 11),                # HipR -> KneeR -> AnkleR
            (3, 5), (5, 8),                 # ShL -> ElL -> WrL
            (2, 4), (4, 7),                 # ShR -> ElR -> WrR
            (0, 3), (6, 2),                 # HipL -> ShL, HipR -> ShR
            (0, 6),                         # HipL -> HipR (Lower torso)
            (3, 2),                         # ShL -> ShR (Upper torso)
            (3, 1), (2, 1)                  # ShL -> Head, ShR -> Head
        ]
    elif pose.shape[0] == 17:
        connections = [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
            (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
        ]
    else:
        connections = [(i, i+1) for i in range(pose.shape[0]-1)]

    for i, j in connections:
        ax.plot([pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]], color=color)
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, label=label)
    
    # Add joint index labels
    for i in range(pose.shape[0]):
        ax.text(pose[i, 0], pose[i, 1], pose[i, 2], str(i), color='black', fontsize=8)

def evaluate(model_pos, test_generator, receptive_field, args, kps_left, kps_right, joints_left, joints_right, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    epoch_pck = 0
    pck_threshold = 150 # mm
    
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for cam, batch, batch_2d in test_generator.next_epoch():
            if cam is not None:
                cam = torch.from_numpy(cam.astype('float32'))
            
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            inputs_3d = torch.from_numpy(batch.astype('float32'))

            # Test-time augmentation (Flip)
            inputs_2d_flip = inputs_2d.clone()
            inputs_2d_flip [:, :, :, 0] *= -1
            inputs_2d_flip[:, :, kps_left + kps_right,:] = inputs_2d_flip[:, :, kps_right + kps_left,:]

            # Prepare Data
            inputs_2d, inputs_3d = eval_data_prepare(receptive_field, inputs_2d, inputs_3d)
            inputs_2d_flip, _ = eval_data_prepare(receptive_field, inputs_2d_flip, inputs_3d)
            
            if cam is not None:
                cam = cam.repeat(inputs_2d.size(0), 1)

            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            inputs_2d = inputs_2d.to(device)
            inputs_2d_flip = inputs_2d_flip.to(device)
            inputs_3d = inputs_3d.to(device)
            if cam is not None:
                cam = cam.to(device)
            
            inputs_3d[:, :, 0] = 0 # Center root

            # Forward Pass
            predicted_3d_pos = model_pos(inputs_2d)
            predicted_3d_pos_flip = model_pos(inputs_2d_flip)

            # Flip back
            predicted_3d_pos_flip[:, :, :, 0] *= -1
            predicted_3d_pos_flip[:, :, joints_left + joints_right] = predicted_3d_pos_flip[:, :, joints_right + joints_left]

            # Average
            predicted_3d_pos = torch.mean(torch.cat((predicted_3d_pos, predicted_3d_pos_flip), dim=1), dim=1, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(1).cpu().numpy(), inputs_3d.squeeze(1).cpu().numpy(), 0, 0, 0

            del inputs_2d, inputs_2d_flip
            torch.cuda.empty_cache()

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            
            scale_factor = 100 if args.dataset == 'dhp19' else 1000
            diff = (predicted_3d_pos - inputs_3d) * scale_factor
            dist = torch.norm(diff, dim=-1) # (N, 1, J)
            epoch_pck += torch.sum(dist < pck_threshold).item()

            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            _, batch_p_mpjpe = p_mpjpe_with_coords(predicted_3d_pos, inputs)
            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * batch_p_mpjpe
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    scale_factor = 100 if args.dataset == 'dhp19' else 1000
    if N == 0:
        if action is not None:
             print(f"DEBUG: No frames processed for {action}.")
        if return_predictions:
            return None, None, 0, 0, 0
        return 0, 0, 0, 0, 0

    e1 = (epoch_loss_3d_pos / N) * scale_factor
    e2 = (epoch_loss_3d_pos_procrustes / N) * scale_factor
    e3 = (epoch_loss_3d_pos_scale / N) * scale_factor
    ev = (epoch_loss_3d_vel / N) * scale_factor
    pck = (epoch_pck / (N * predicted_3d_pos.shape[-2])) * 100

    if action is not None:
        print(f"Sequence Evaluation: {action}")
        print(f"  MPJPE: {e1:.2f} mm")
        print(f"  P-MPJPE: {e2:.2f} mm")
        print('----------')

    return e1, e2, e3, ev, pck

def main():
    # Use additional parser to extract custom arguments
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument('--csv_file', type=str, help='Path to 2D keypoints CSV')
    extra_parser.add_argument('--batch_dir_movenet', type=str, help='Path to directory containing movenet.csv subdirectories (e.g. cam0_S13_1_1)')
    extra_args, remaining = extra_parser.parse_known_args()

    # Trick parse_args() from PoseFormerV2 to only see the remaining arguments
    import sys
    orig_argv = sys.argv
    sys.argv = [orig_argv[0]] + remaining
    args = parse_args()
    sys.argv = orig_argv # Restore

    if extra_args.csv_file:
        args.csv_file = extra_args.csv_file
    else:
        args.csv_file = None
        
    if extra_args.batch_dir_movenet:
        args.batch_dir_movenet = extra_args.batch_dir_movenet
    else:
        args.batch_dir_movenet = None
    
    # Force DHP19 settings if not provided
    if args.dataset == 'dhp19':
        print("Using DHP19 configuration.")
    else:
        print(f"Warning: Dataset argument is {args.dataset}, expecting 'dhp19' for this specific setup.")

    # Symmetry for DHP19
    joints_left = [0, 3, 5, 8, 10, 12]
    joints_right = [6, 2, 4, 7, 9, 11]
    
    # 2D Symmetry for MoveNet-13
    kps_left = [1, 3, 5, 7, 9, 11]
    kps_right = [2, 4, 6, 8, 10, 12]

    # 1. Load Dataset
    print('Loading dataset...')
    target_path = 'DHP19/fixed_gt/data_3d_dhp19.npz'
    if not os.path.exists(target_path):
        target_path = 'data/data_3d_dhp19.npz'
    if not os.path.exists(target_path):
        target_path = 'PoseFormerV2-main/data/data_3d_dhp19.npz'
        
    print(f"Loading 3D data from: {target_path}")
    dataset = Dhp19Dataset(target_path)
    print(f"DEBUG: Available subjects in 3D dataset: {list(dataset.subjects())}")
    
    # 2. Process Data (Normalize /100 and Transform to Camera Space)
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions_3d' in anim:
                positions_3d_list = anim['positions_3d']
                for i, pos_3d in enumerate(positions_3d_list):
                    # 1. Transform from World to Camera Space
                    if subject in dataset.cameras():
                        cams = dataset.cameras()[subject]
                        if i < len(cams) and 'R' in cams[i] and 't' in cams[i]:
                            # world_to_camera expects (N, J, 3) and R (quaternion style usually in PoseFormer)
                            # DHP19 common/camera.py implementation for world_to_camera:
                            # def world_to_camera(X, R, t):
                            #     Rt = wrap(qinverse, R)
                            #     return wrap(qrot, np.tile(Rt, (*X.shape[:-1], 1)), X - t)
                            pos_3d = world_to_camera(pos_3d, cams[i]['R'], cams[i]['t'])
                    
                    # 2. Root-relative coordinates (subtract root from all joints)
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    
                    # 3. Normalize to Decimeters (matching model convention)
                    pos_3d_normalized = pos_3d.copy()
                    pos_3d_normalized = pos_3d_normalized / 100.0
                    positions_3d_list[i] = pos_3d_normalized
                anim['positions_3d'] = positions_3d_list

    # 3. Load 2D Detections
    # MoveNet to DHP19 Joint Mapping
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
        12: 12  # AnR -> AnL (missing index in request but standard for DHP19)
    }

    keypoints = {}
    
    if args.batch_dir_movenet:
        print(f'Batch loading 2D detections from directory: {args.batch_dir_movenet}')
        if not os.path.exists(args.batch_dir_movenet):
            print(f"Error: Directory {args.batch_dir_movenet} does not exist.")
            return

        for subdir_name in os.listdir(args.batch_dir_movenet):
            subdir_path = os.path.join(args.batch_dir_movenet, subdir_name)
            if not os.path.isdir(subdir_path):
                continue

            # Expected format: camX_SXX_X_X
            try:
                parts = subdir_name.split('_')
                if len(parts) >= 4 and parts[0].startswith('cam'):
                    cam_idx = int(parts[0].replace('cam', ''))
                    # Usually: S13, 1, 1 - reassemble the action string
                    # e.g., 'S13' and action '1_1'
                    subject = parts[1]
                    action = f"{parts[2]}_{parts[3]}"
                else:
                    # Skip if it doesn't match expected pattern
                    continue
            except ValueError:
                continue

            input_csv = os.path.join(subdir_path, 'movenet.csv')
            output_csv = os.path.join(subdir_path, 'moveenet.csv')

            if not os.path.exists(input_csv) and not os.path.exists(output_csv):
                print(f"Skipping {subdir_name}: neither movenet.csv nor moveenet.csv found.")
                continue

            # 1. Format the CSV
            if not os.path.exists(output_csv):
                format_movenet_csv(input_csv, output_csv)
            else:
                print(f"DEBUG: Using existing formatted CSV: {output_csv}")

            # 2. Load the formatted predictions
            kps_2d = load_2d_predictions(output_csv, num_joints=13, mapping=movenet_to_dhp19)
            
            # Apply Y-flip (260 - y) for DHP19
            kps_2d[..., 1] = 260 - kps_2d[..., 1]
            
            # Center at root (DHP19 index 0)
            kps_2d = kps_2d - kps_2d[:, 0:1, :]

            # Build dict
            if subject not in keypoints:
                keypoints[subject] = {}
            if action not in keypoints[subject]:
                keypoints[subject][action] = [None, None, None, None]
                
            keypoints[subject][action][cam_idx] = kps_2d
            print(f"DEBUG: Successfully loaded {subject} {action} cam {cam_idx} with shape {kps_2d.shape}")

    elif args.csv_file:
        print(f'Loading single 2D detection from CSV: {args.csv_file}')
        subject = args.viz_subject if args.viz_subject else 'S13'
        action = args.viz_action if args.viz_action else '1_1'
        cam_idx = args.viz_camera # default 0
        
        # Load with mapping
        kps_2d = load_2d_predictions(args.csv_file, num_joints=13, mapping=movenet_to_dhp19)
        
        # Apply Y-flip (260 - y) for DHP19
        print("Applying Y-axis flip (260 - y)...")
        kps_2d[..., 1] = 260 - kps_2d[..., 1]
        
        # Center at root (DHP19 index 0)
        print("Centering 2D inputs at root (DHP19 index 0)...")
        kps_2d = kps_2d - kps_2d[:, 0:1, :]
        
        # Initialize dictionary with 4 empty camera slots
        keypoints = {subject: {action: [None, None, None, None]}}
        keypoints[subject][action][cam_idx] = kps_2d
        print(f"DEBUG: Manually built keypoints for {subject} {action} cam {cam_idx}")
    else:
        print('Error: No data inputs provided. Use --csv_file or --batch_dir_movenet.')
        return

    # 4. Normalize 2D
    for subject in keypoints.keys():
        for action in keypoints[subject].keys():
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                if kps is None:
                    continue
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    # 5. Model Setup
    print("DEBUG: Setting up PoseTransformerV2...")
    receptive_field = args.number_of_frames
    num_joints = 13  # or use dataset metadata for num_joints
    
    model = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args)

    # # Model architecture settings to match fine-tuned H36M checkpoint
    # model_num_joints = 17
    # num_joints_out = 13
    # model_embed_dim = 544
    # joint_mapping = [10, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
    
    # model = PoseTransformerV2(num_frame=receptive_field, num_joints=model_num_joints, in_chans=2, embed_dim=model_embed_dim, joint_mapping=joint_mapping,
    #     num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args, num_joints_out=num_joints_out)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = nn.DataParallel(model)
    model = model.to(device)
    
    checkpoint_path = os.path.join(args.checkpoint, args.evaluate) if args.evaluate else args.checkpoint
    if os.path.isfile(args.checkpoint):
        checkpoint_path = args.checkpoint
        
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    model.eval()

    pad = (receptive_field - 1) // 2
    causal_shift = 0

    # 6. Evaluation/Visualization Loop
    viz_subject = args.viz_subject
    viz_action = args.viz_action
    
    if viz_subject and viz_action:
        print(f"\nDEBUG: Visualizing specific sequence: {viz_subject} {viz_action}")
        cameras_act, poses_act, poses_2d_act = fetch(dataset, keypoints, args, [viz_subject], action_filter=[viz_action])
        if poses_act is None:
            print(f"DEBUG: poses_act is None for {viz_subject} {viz_action}")
            print("Error: Could not find specified sequence for visualization.")
            return
        
        print(f"DEBUG: Sequence found. Frames: {poses_act[0].shape[0] if poses_act else 0}")
            
        gen = UnchunkedGenerator(cameras_act, poses_act, poses_2d_act,
                                 pad=pad, causal_shift=causal_shift, augment=False,
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)
        
        print("DEBUG: Calling evaluate() to get predictions and sequence metrics...")
        # Get sequence-wide metrics first
        e1_seq, e2_seq, _, _, _ = evaluate(model, gen, receptive_field, args, kps_left, kps_right, joints_left, joints_right, 
                                action=f"{viz_subject}_{viz_action}_seq", return_predictions=False)
        
        # Then get predictions for plotting
        preds, targs, _, _, _ = evaluate(model, gen, receptive_field, args, kps_left, kps_right, joints_left, joints_right, 
                                action=None, return_predictions=True)
        
        if preds is None:
            print("\nERROR: No predictions returned. Possible causes:")
            print("1. The sequence is shorter than the receptive field (default 27 frames).")
            print("2. All frames in the sequence contain NaNs and were filtered out.")
            return

        print(f"DEBUG: evaluate() returned. preds shape: {preds.shape}")
        
        # Scale Correction for Predicted Poses
        # If mean height is small (< 5), then the model likely predicts in meters (PoseFormer default)
        # while GT is in decimeters (mm/100). We normalize both to decimeters for internal consistency.
        pred_mean_size = np.mean(np.linalg.norm(preds, axis=-1))
        targ_mean_size = np.mean(np.linalg.norm(targs, axis=-1))
        
        if pred_mean_size < (targ_mean_size / 5.0):
            print(f"INFO: Prediction scale ({pred_mean_size:.2f}) appears to be in meters. Scaling up by 10x to match decimeters.")
            preds = preds * 10.0
            
        # Calculate MPJPE for this visualized sample (scaled back to mm)
        scale_val = (100.0 if args.dataset == 'dhp19' else 1000.0)
        error = np.mean(np.linalg.norm(preds - targs, axis=-1)) * scale_val
        
        # Calculate P-MPJPE for the entire sequence
        # We pass the full (N, J, 3) arrays to p_mpjpe_with_coords
        _, seq_p_mpjpe = p_mpjpe_with_coords(preds * scale_val, targs * scale_val)
        
        print(f"\nEvaluation Results for {viz_subject} {viz_action}:")
        print(f"Protocol #1 Error (MPJPE): {error:.2f} mm")
        print(f"Protocol #2 Error (P-MPJPE): {seq_p_mpjpe:.2f} mm")
        
        # Plot a single frame
        frame_idx = min(100, preds.shape[0] - 1)
        pred_frame = np.squeeze(preds[frame_idx])
        targ_frame = np.squeeze(targs[frame_idx])
        
        # Center at root (joint 0)
        pred_frame = pred_frame - pred_frame[0:1, :]
        targ_frame = targ_frame - targ_frame[0:1, :]

        # Scale back to mm for reasonable plot units
        scale_to_mm = 100.0 if args.dataset == 'dhp19' else 1000.0
        pred_frame *= scale_to_mm
        targ_frame *= scale_to_mm
        
        # Procrustes Alignment for Visualization
        print("Calculating Procrustes alignment for visualization...")
        # target (target), source (predicted)
        aligned_pred_frame, p_mpjpe_val = p_mpjpe_with_coords(pred_frame[None, ...], targ_frame[None, ...])
        aligned_pred_frame = np.squeeze(aligned_pred_frame)
        print("aligned_pred_frame",aligned_pred_frame.shape, aligned_pred_frame[0:13])
        print("targ_frame",targ_frame.shape, targ_frame[0:13])
        print(f"P-MPJPE (Visualization Frame {frame_idx} Only): {p_mpjpe_val:.2f} mm")
        
        # Show only GT and Aligned Prediction (labeled as Prediction)
        show_3d_pose(aligned_pred_frame, targ_frame, frame_idx)
        return

    # Otherwise, run batch evaluation as before
    subjects_test = list(keypoints.keys())
        
    actions_filter = args.actions.split(',') if args.actions != '*' else None

    # Collect actions
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}
        for action in dataset[subject].keys():
            # Actions in Dhp19Dataset are usually '1_1', '1_2', etc.
            action_name = action
            if actions_filter is not None:
                if not any(action.startswith(af) for af in actions_filter):
                    continue
            if action_name not in all_actions_by_subject[subject]:
                all_actions_by_subject[subject][action_name] = []
            all_actions_by_subject[subject][action_name].append((subject, action))

    print('\nEvaluating Batch...')
    results = []
    
    for subject in all_actions_by_subject.keys():
        for action_key, actions in all_actions_by_subject[subject].items():
            cameras_act, poses_act, poses_2d_act = fetch(dataset, keypoints, args, [subject], action_filter=[action_key])
            if poses_act is None:
                continue
            
            # Now evaluate each camera sequence individually
            for i in range(len(poses_act)):
                cam_id = i # Simple ID
                
                # Single sequence generator
                seq_cameras = [cameras_act[i]] if cameras_act is not None else None
                gen = UnchunkedGenerator(seq_cameras, [poses_act[i]], [poses_2d_act[i]],
                                         pad=pad, causal_shift=causal_shift, augment=False,
                                         kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                         joints_right=joints_right)
                
                label = f"{subject}_{action_key}_cam{cam_id}"
                e1, e2, e3, ev, pck = evaluate(model, gen, receptive_field, args, kps_left, kps_right, joints_left, joints_right, action=label)
                
                if e2 is not None and e2 > 0:
                    results.append({
                        'label': label,
                        'subject': subject,
                        'action': action_key,
                        'camera': cam_id,
                        'p_mpjpe': e2
                    })

    if results:
        print("\n" + "="*60)
        print(f"{'Sequence':<40} | {'P-MPJPE (mm)':<12}")
        print("-" * 55)
        all_p_mpjpes = [r['p_mpjpe'] for r in results]
        
        for r in results:
            print(f"{r['label']:<40} | {r['p_mpjpe']:>12.2f}")
        
        print("-" * 55)
        print(f"{'AVERAGE':<40} | {np.mean(all_p_mpjpes):>12.2f}")
        print("="*60)
    else:
        print("\nNo sequences were evaluated (check your filters).")

if __name__ == "__main__":
    main()
