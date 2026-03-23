import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def show_skeleton(pose_3d, kps_2d, title):
    fig = plt.figure(figsize=(14, 7))
    
    # --- 3D Plot ---
    ax = fig.add_subplot(121, projection='3d')
    
    # connections for DHP19 13-joint
    connections = [
        (0, 10), (10, 12),    # Left Leg: hipL -> kneeL -> footL
        (6, 9), (9, 11),      # Right Leg: hipR -> kneeR -> footR
        (3, 5), (5, 8),       # Left Arm: shoL -> elbL -> handL
        (2, 4), (4, 7),       # Right Arm: shoR -> elbR -> handR
        (0, 6),               # Hips: hipL -> hipR
        (3, 2),               # Shoulders: shoL -> shoR
        (0, 3), (6, 2),       # Torso
        (3, 1), (2, 1)        # Head
    ]
    
    for i, j in connections:
        ax.plot([pose_3d[i, 0], pose_3d[j, 0]],
                [pose_3d[i, 1], pose_3d[j, 1]],
                [pose_3d[i, 2], pose_3d[j, 2]], color='blue')

    ax.scatter(pose_3d[:, 0], pose_3d[:, 1], pose_3d[:, 2], c='red', label='Joints')
    # Highlight HipL (idx 0)
    ax.scatter(pose_3d[0, 0], pose_3d[0, 1], pose_3d[0, 2], c='green', s=100, label='HipL (Idx 0)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"{title} (3D)")
    ax.legend()
    
    # Set equal aspect ratio
    mid_x = np.mean(pose_3d[:, 0])
    mid_y = np.mean(pose_3d[:, 1])
    mid_z = np.mean(pose_3d[:, 2])
    radius = 1000 
    ax.set_xlim3d([mid_x - radius, mid_x + radius])
    ax.set_ylim3d([mid_y - radius, mid_y + radius])
    ax.set_zlim3d([mid_z - radius, mid_z + radius])
    
    # --- 2D Plot ---
    ax2 = fig.add_subplot(122)
    # kps_2d shape: (13, 2)
    
    for i, j in connections:
        ax2.plot([kps_2d[i, 0], kps_2d[j, 0]],
                 [kps_2d[i, 1], kps_2d[j, 1]], color='blue')
                 
    ax2.scatter(kps_2d[:, 0], kps_2d[:, 1], c='red', label='Joints')
    ax2.scatter(kps_2d[0, 0], kps_2d[0, 1], c='green', s=100, label='HipL')
    
    ax2.set_xlabel('U (pixels)')
    ax2.set_ylabel('V (pixels)')
    ax2.set_title("2D Projection (Camera 2)")
    # ax2.invert_yaxis() # User reported this makes it upside down, so disabling it.
    ax2.set_aspect('equal')
    
    plt.show()

def main():
    path_3d = os.path.join('DHP19/fixed_gt', 'data_3d_dhp19.npz')
    path_2d = os.path.join('DHP19/fixed_gt', 'data_2d_dhp19_gt.npz')

    
    data_3d = np.load(path_3d, allow_pickle=True)['positions_3d'].item()
    data_2d = np.load(path_2d, allow_pickle=True)['positions_2d'].item()
    
    subject = 'S10'
    action_key = '3_1' 
    
    if subject not in data_3d:
        subject = list(data_3d.keys())[0]
        
    # Find action
    final_action = None
    for k in data_3d[subject].keys():
        if k.startswith(action_key):
            final_action = k
            break
            
    if not final_action:
        final_action = list(data_3d[subject].keys())[0]

    print(f"Visualizing Subject: {subject}, Action: {final_action}")
    
    # Get 3D sequence
    pos_3d_seq = data_3d[subject][final_action]
    if pos_3d_seq.ndim == 2:
         pos_3d_seq = pos_3d_seq[np.newaxis, :, :]

    # Get 2D sequence (Camera 2 -> idx 1)
    kps_2d_seq = data_2d[subject][final_action][1] 
    if kps_2d_seq.ndim == 2:
        kps_2d_seq = kps_2d_seq[np.newaxis, :, :]

    # Frame 0
    pose_3d = pos_3d_seq[0]
    pose_2d = kps_2d_seq[0]
    
    print("Raw HipL 3D (idx 0):", pose_3d[0])
    print("Raw HipL 2D (idx 0):", pose_2d[0])
    
    show_skeleton(pose_3d, pose_2d, f"Raw GT Frame 0")

if __name__ == "__main__":
    main()
