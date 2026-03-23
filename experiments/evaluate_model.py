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

def fetch(dataset, keypoints, args, subjects, action_filter=None, subset=1, parse_3d_poses=True, load_gt=False):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    
    for subject in subjects:
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
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                
                for i in range(len(poses_3d)): # Iterate across cameras
                    # Check for NaNs in this sequence
                    valid_3d = not np.isnan(poses_3d[i]).any()
                    valid_2d = not np.isnan(poses_2d[i]).any()
                    
                    if not valid_3d or not valid_2d:
                        # print(f"WARNING: NaNs found in {subject} {action} camera {i}. Skipping sequence.")
                        continue
                        
                    out_poses_3d.append(poses_3d[i])
                    out_poses_2d.append(poses_2d[i])
                    
                    # Also append camera params only if sequence is valid
                    if subject in dataset.cameras():
                         cams = dataset.cameras()[subject]
                         if 'intrinsic' in cams[i]:
                             out_camera_params.append(cams[i]['intrinsic'])
            else:
                # If NOT parsing 3D poses
                for i in range(len(poses_2d)):
                     if np.isnan(poses_2d[i]).any():
                         continue 
                     out_poses_2d.append(poses_2d[i])
                     if subject in dataset.cameras():
                         cams = dataset.cameras()[subject]
                         if 'intrinsic' in cams[i]:
                             out_camera_params.append(cams[i]['intrinsic'])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        print(f"DEBUG: fetch() found no valid sequences for {subjects} {action_filter} (checked for NaNs)")
        out_poses_3d = None

    # Stride/Downsample logic from original script
    stride = args.downsample
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i])//stride * subset)*stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames:stride]
            if out_poses_3d is not None:
                if i < len(out_poses_3d):
                    out_poses_3d[i] = out_poses_3d[i][start:start+n_frames:stride]
    elif stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                if i < len(out_poses_3d):
                    out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d

def eval_data_prepare(receptive_field, inputs_2d, inputs_3d):
    inputs_2d_p = torch.squeeze(inputs_2d)
    inputs_3d_p = inputs_3d.permute(1,0,2,3)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d, inputs_3d_p

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
    _plot_single_pose(ax, predicted, 'red', 'Predicted')
    
    ax.set_title(f'Frame {frame_idx} Overlay\n(Blue=GT, Red=Pred)')
    ax.legend()
    
    # Set limits
    all_pts = np.concatenate([predicted, target], axis=0)
    center = np.mean(all_pts, axis=0)
    radius = np.max(np.linalg.norm(all_pts - center, axis=1)) * 1.1 # Dynamic scale with 10% margin
    if radius < 1.0: radius = 1000.0 # Fallback for mm units
    
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])
    
    # Aspect ratio should be equal
    ax.set_box_aspect([1, 1, 1])
    
    print(f"Saving 3D plot to evaluate_viz.png...")
    plt.savefig('evaluate_viz.png')
    print(f"Showing 3D plot for frame {frame_idx}... Close the window to continue.")
    plt.show()

def _plot_single_pose(ax, pose, color, label):
    if pose.shape[0] == 13:
        # DHP19 joints
        connections = [
            (0, 10), (10, 12), (6, 9), (9, 11), (3, 5), (5, 8), (2, 4), (4, 7),
            (0, 6), (3, 2), (0, 3), (6, 2), (3, 1), (2, 1)
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
        # print(f"DEBUG: evaluate() loop starting for {action if action else 'all'} sequences...")
        for cam, batch, batch_2d in test_generator.next_epoch():
            # print(f"DEBUG: Generator yielded a sequence. Shape: {batch.shape}")
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
                # Return predictions and targets for this batch
                return predicted_3d_pos.squeeze(1).cpu().numpy(), inputs_3d.squeeze(1).cpu().numpy(), 0, 0, 0

            del inputs_2d, inputs_2d_flip
            torch.cuda.empty_cache()

            # Calculate Error
            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0]*inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos, inputs_3d).item()
            epoch_loss_3d_pos += inputs_3d.shape[0]*inputs_3d.shape[1] * error.item()
            
            scale_factor = 100 if args.dataset == 'dhp19' else 1000
            # Calculate PCK
            # predicted_3d_pos: (N, 1, J, 3), inputs_3d: (N, 1, J, 3)
            diff = (predicted_3d_pos - inputs_3d) * scale_factor
            dist = torch.norm(diff, dim=-1) # (N, 1, J)
            epoch_pck += torch.sum(dist < pck_threshold).item()

            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0]*inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)
            epoch_loss_3d_vel += inputs_3d.shape[0]*inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    scale_factor = 100 if args.dataset == 'dhp19' else 1000
    if N == 0:
        if action is not None:
             print(f"DEBUG: No frames processed for {action}.")
        if return_predictions:
            return None, None, 0, 0, 0
        return 0, 0, 0, 0, 0

    e1 = (epoch_loss_3d_pos / N)*scale_factor
    e2 = (epoch_loss_3d_pos_procrustes / N)*scale_factor
    e3 = (epoch_loss_3d_pos_scale / N)*scale_factor
    ev = (epoch_loss_3d_vel / N)*scale_factor
    pck = (epoch_pck / (N * predicted_3d_pos.shape[-2])) * 100

    if action is not None:
        print(f"Evaluating {action}")
        print(f" MPJPE: {e1:.2f} mm")
        print(f" PCK @ {pck_threshold:.0f}mm: {pck:.2f}%")
        print('----------')

    return e1, e2, e3, ev, pck


def main():
    # Simulate arguments or parse them if needed. 
    # For standalone, we can hardcode or re-use parse_args but override specific things.
    args = parse_args()
    
    # Force DHP19 settings if not provided
    if args.dataset == 'dhp19':
        print("Using DHP19 configuration.")
    else:
        print(f"Warning: Dataset argument is {args.dataset}, expecting 'dhp19' for this specific setup.")

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
    
    # 2. Process Data (Normalize /100)
    for subject in dataset.subjects():
        for action in dataset[subject].keys():
            anim = dataset[subject][action]
            if 'positions_3d' in anim:
                positions_3d_list = anim['positions_3d']
                for i, pos_3d in enumerate(positions_3d_list):
                    # Ensure root-relative coordinates (subtract root from all joints)
                    pos_3d[:, 1:] -= pos_3d[:, :1]
                    
                    pos_3d_normalized = pos_3d.copy()
                    pos_3d_normalized = pos_3d_normalized / 100.0
                    positions_3d_list[i] = pos_3d_normalized
                anim['positions_3d'] = positions_3d_list

    # 3. Load 2D Detections
    print('Loading 2D detections...')
    keypoints_path = 'DHP19/fixed_gt/data_2d_dhp19_gt.npz'
    if not os.path.exists(keypoints_path):
        keypoints_path = 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'
    if not os.path.exists(keypoints_path):
        keypoints_path = 'PoseFormerV2-main/data/data_2d_dhp19_gt.npz'
        
    print(f"Loading 2D data from: {keypoints_path}")
    keypoints = np.load(keypoints_path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    # 4. Normalize 2D
    for subject in keypoints.keys():
        for action in keypoints[subject]:
            for cam_idx, kps in enumerate(keypoints[subject][action]):
                cam = dataset.cameras()[subject][cam_idx]
                kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                keypoints[subject][action][cam_idx] = kps

    # 5. Model Setup
    print("DEBUG: Setting up PoseTransformerV2...")
    receptive_field = args.number_of_frames
    num_joints = keypoints_metadata['num_joints']
    
    model = PoseTransformerV2(num_frame=receptive_field, num_joints=num_joints, in_chans=2,
        num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0, args=args)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = nn.DataParallel(model)
    model = model.to(device)
    
    checkpoint_path = os.path.join(args.checkpoint, args.evaluate) if args.evaluate else args.checkpoint
    # If args.checkpoint is a file, use it directly
    if os.path.isfile(args.checkpoint):
        checkpoint_path = args.checkpoint
        
    print(f"Loading checkpoint: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older torch versions without weights_only
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
    model.load_state_dict(checkpoint['model_pos'], strict=False)
    model.eval()

    pad = (receptive_field - 1) // 2
    causal_shift = 0

    # 6. Evaluation/Visualization Loop
    # Users can filter subjects/actions via --subjects-test and --actions (or --viz-subject/--viz-action)
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
        
        print("DEBUG: Calling evaluate() to get predictions...")
        preds, targs, _, _, pck = evaluate(model, gen, receptive_field, args, kps_left, kps_right, joints_left, joints_right, 
                                action=viz_action, return_predictions=True)
        
        if preds is None:
            print("\nERROR: No predictions returned. Possible causes:")
            print("1. The sequence is shorter than the receptive field (default 27 frames).")
            print("2. All frames in the sequence contain NaNs and were filtered out.")
            return

        print(f"DEBUG: evaluate() returned. preds shape: {preds.shape}")
        
        # Calculate MPJPE for this visualized sample (scaled back to mm)
        # preds: (Frames, Joints, 3), targs: (Frames, Joints, 3)
        # Note: Both are already root-centered in evaluate()
        error = np.mean(np.linalg.norm(preds - targs, axis=-1)) * (100.0 if args.dataset == 'dhp19' else 1000.0)
        print(f"\nEvaluation Results for {viz_subject} {viz_action}:")
        print(f"Protocol #1 Error (MPJPE): {error:.2f} mm")
        
        # Plot a single frame
        frame_idx = min(100, preds.shape[0] - 1)
        # Ensure 2D (Joints, 3) and center at root
        pred_frame = np.squeeze(preds[frame_idx])
        targ_frame = np.squeeze(targs[frame_idx])
        
        pred_frame = pred_frame - pred_frame[0:1, :]
        targ_frame = targ_frame - targ_frame[0:1, :]
        
        # Scale back to mm for reasonable plot units (DHP19 training used /100)
        pred_frame *= 100.0
        targ_frame *= 100.0
        
        show_3d_pose(pred_frame, targ_frame, frame_idx)
        return

    # Otherwise, run batch evaluation as before
    subjects_test = args.subjects_test.split(',')
    if args.dataset == 'dhp19':
        subjects_test = ['S13', 'S14', 'S15', 'S16', 'S17']
        print(f"INFO: Using default DHP19 test set: {subjects_test}")
        
    actions_filter = args.actions.split(',') if args.actions != '*' else None

    # Collect actions
    all_actions_by_subject = {}
    for subject in subjects_test:
        if subject not in all_actions_by_subject:
            all_actions_by_subject[subject] = {}
        for action in dataset[subject].keys():
            action_name = action.split(' ')[0]
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
                
                if e1 is not None and e1 > 0:
                    results.append({
                        'label': label,
                        'subject': subject,
                        'action': action_key,
                        'camera': cam_id,
                        'mpjpe': e1,
                        'pck': pck
                    })

    if results:
        print("\n" + "="*60)
        print(f"{'Sequence':<40} | {'MPJPE (mm)':<12} | {'PCK (%)':<8}")
        print("-" * 65)
        all_mpjpes = [r['mpjpe'] for r in results]
        all_pcks = [r['pck'] for r in results]
        
        for r in results:
            print(f"{r['label']:<40} | {r['mpjpe']:>12.2f} | {r['pck']:>8.2f}")
        
        print("-" * 65)
        print(f"{'AVERAGE':<40} | {np.mean(all_mpjpes):>12.2f} | {np.mean(all_pcks):>8.2f}")
        print("="*60)
    else:
        print("\nNo sequences were evaluated (check your filters).")

if __name__ == "__main__":
    main()
