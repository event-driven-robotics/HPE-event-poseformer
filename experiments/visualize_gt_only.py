import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add current directory to path to allow imports from common
sys.path.append(os.path.join(os.getcwd(), 'PoseFormerV2-main'))

from common.arguments import parse_args
from common.dhp19_dataset import Dhp19Dataset
from common.utils import deterministic_random

def fetch(dataset, keypoints, args, subjects, action_filter=None, subset=1, parse_3d_poses=True):
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
                        continue
                        
                    out_poses_3d.append(poses_3d[i])
                    out_poses_2d.append(poses_2d[i])
                    
                    if subject in dataset.cameras():
                         cams = dataset.cameras()[subject]
                         if 'intrinsic' in cams[i]:
                             out_camera_params.append(cams[i]['intrinsic'])
            else:
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
        out_poses_3d = None

    # Stride/Downsample logic
    stride = args.downsample
    if stride > 1:
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d

def _get_connections(num_joints):
    if num_joints == 13:
        # DHP19 joints
        return [
            (0, 10), (10, 12), (6, 9), (9, 11), (3, 5), (5, 8), (2, 4), (4, 7),
            (0, 6), (3, 2), (0, 3), (6, 2), (3, 1), (2, 1)
        ]
    elif num_joints == 17:
        # H36M joints
        return [
            (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (0, 7), (7, 8),
            (8, 9), (9, 10), (8, 11), (11, 12), (12, 13), (8, 14), (14, 15), (15, 16)
        ]
    else:
        return [(i, i+1) for i in range(num_joints-1)]

def _plot_2d_pose(ax, pose, color, label):
    connections = _get_connections(pose.shape[0])
    for i, j in connections:
        ax.plot([pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]], color=color)
    ax.scatter(pose[:, 0], pose[:, 1], c=color, label=label, s=20)
    ax.set_aspect('equal')
    # Invert Y for image coordinates
    # ax.invert_yaxis()

def _plot_3d_pose(ax, pose, color, label):
    connections = _get_connections(pose.shape[0])
    for i, j in connections:
        ax.plot([pose[i, 0], pose[j, 0]],
                [pose[i, 1], pose[j, 1]],
                [pose[i, 2], pose[j, 2]], color=color)
    ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c=color, label=label, s=20)

def show_2d_3d_gt(pose_2d, pose_3d, frame_idx, title=""):
    fig = plt.figure(figsize=(15, 7))
    
    # 2D Plot
    ax2d = fig.add_subplot(121)
    _plot_2d_pose(ax2d, pose_2d, 'green', 'GT 2D')
    ax2d.set_title(f"2D Ground Truth (Frame {frame_idx})")
    ax2d.legend()
    
    # 3D Plot
    ax3d = fig.add_subplot(122, projection='3d')
    _plot_3d_pose(ax3d, pose_3d, 'blue', 'GT 3D')
    
    # Dynamic Scaling for 3D
    center = np.mean(pose_3d, axis=0)
    radius = np.max(np.linalg.norm(pose_3d - center, axis=1)) * 1.1
    if radius < 1.0: radius = 1000.0 # Fallback for mm
    
    ax3d.set_xlim3d([center[0] - radius, center[0] + radius])
    ax3d.set_ylim3d([center[1] - radius, center[1] + radius])
    ax3d.set_zlim3d([center[2] - radius, center[2] + radius])
    ax3d.set_box_aspect([1, 1, 1])
    ax3d.set_title(f"3D Ground Truth (Frame {frame_idx})")
    ax3d.legend()
    
    fig.suptitle(title)
    
    save_path = f"gt_viz_{frame_idx}.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")
    plt.show()

def main():
    args = parse_args()
    
    # 1. load 3D dataset
    target_path = 'data/data_3d_dhp19.npz'
    if not os.path.exists(target_path):
        target_path = 'PoseFormerV2-main/data/data_3d_dhp19.npz'
    
    print(f"Loading 3D data from: {target_path}")
    dataset = Dhp19Dataset(target_path)

    # 2. load 2D detections
    keypoints_path = 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'
    if not os.path.exists(keypoints_path):
        keypoints_path = 'PoseFormerV2-main/data/data_2d_dhp19_gt.npz'
        
    print(f"Loading 2D data from: {keypoints_path}")
    keypoints = np.load(keypoints_path, allow_pickle=True)
    keypoints = keypoints['positions_2d'].item()

    viz_subject = args.viz_subject
    viz_action = args.viz_action
    
    if not viz_subject or not viz_action:
        # Default to some sequence if not provided
        viz_subject = 'S14'
        viz_action = '4_1'
        print(f"No subject/action provided. Using default: {viz_subject} {viz_action}")

    print(f"DEBUG: Visualizing GT for: {viz_subject} {viz_action}")
    cameras_act, poses_3d_act, poses_2d_act = fetch(dataset, keypoints, args, [viz_subject], action_filter=[viz_action])
    
    if poses_3d_act is None:
        print(f"Error: Could not find specified sequence {viz_subject} {viz_action}")
        return

    # Choose a frame (middle of the sequence)
    idx = 1 # First camera sequence found
    frames = poses_3d_act[idx]
    frames_2d = poses_2d_act[idx]
    frame_idx = min(100, len(frames) - 1)
    
    pose_3d = frames[frame_idx]
    pose_2d = frames_2d[frame_idx]
    
    # Center 3D at root for easier viewing
    pose_3d = pose_3d - pose_3d[0:1, :]
    
    show_2d_3d_gt(pose_2d, pose_3d, frame_idx, title=f"{viz_subject} {viz_action}")

if __name__ == "__main__":
    main()
