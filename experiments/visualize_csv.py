import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def calculate_mpjpe(csv_kps, gt_kps, perfect=False):
    if csv_kps is None or gt_kps is None:
        return None, None
        
    num_frames = min(len(csv_kps), len(gt_kps))
    num_joints = 13
    errors = np.zeros((num_frames, num_joints))
    
    if perfect:
        # CSV is already in DHP19 format and flipped to Y-up
        for f in range(num_frames):
            for j in range(num_joints):
                if j < csv_kps.shape[1] and j < gt_kps.shape[1]:
                    csv_pt = csv_kps[f, j]
                    gt_pt = gt_kps[f, j] # GT is already Y-up
                    errors[f, j] = np.linalg.norm(csv_pt - gt_pt)
    else:
        # Standard MoveEnet mapping
        movenet_to_dhp19 = {
            0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 0, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12
        }
        for f in range(num_frames):
            for mov_idx, dhp_idx in movenet_to_dhp19.items():
                if mov_idx < csv_kps.shape[1] and dhp_idx < gt_kps.shape[1]:
                    csv_pt = csv_kps[f, mov_idx].copy()
                    csv_pt[1] = 260.0 - csv_pt[1] # Flip MoveEnet to Y-up
                    gt_pt = gt_kps[f, dhp_idx]     # GT is already Y-up
                    errors[f, dhp_idx] = np.linalg.norm(csv_pt - gt_pt)
                
    frame_mpjpe = np.mean(errors, axis=1)
    overall_mpjpe = np.mean(frame_mpjpe)
    
    return frame_mpjpe, overall_mpjpe

def save_perfect_csv(csv_kps, output_path):
    if csv_kps is None:
        return
        
    movenet_to_dhp19 = {
        0: 1,   # head -> head
        1: 2,   # sR -> sR
        2: 3,   # sL -> sL
        3: 4,   # eR -> eR
        4: 5,   # eL -> eL
        5: 0,   # hL -> hipL (ROOT-ish)
        6: 6,   # hR -> hipR
        7: 7,   # wR -> wR
        8: 8,   # wL -> wL
        9: 9,   # kR -> kR
        10: 10, # kL -> kL
        11: 11, # aR -> aR
        12: 12  # aL -> aL
    }
    
    # Invert mapping to go from DHP19 index (0-12) to MoveEnet index
    dhp19_to_movenet = {v: k for k, v in movenet_to_dhp19.items()}
    
    num_frames = len(csv_kps)
    num_joints = 13
    
    with open(output_path, 'w') as f:
        f.write("frame,joint,x,y\n")
        for frame_idx in range(num_frames):
            for dhp_idx in range(num_joints):
                if dhp_idx in dhp19_to_movenet:
                    mov_idx = dhp19_to_movenet[dhp_idx]
                    if mov_idx < csv_kps.shape[1]:
                        x, y = csv_kps[frame_idx, mov_idx]
                        # Flip MoveEnet (Y-down) to Perfect (Y-up) to match GT
                        y_flipped = 260.0 - y
                        f.write(f"{frame_idx},{dhp_idx},{x:.2f},{y_flipped:.2f}\n")
                else:
                    # If any joint is missing, write zero or skip
                    f.write(f"{frame_idx},{dhp_idx},0.00,0.00\n")
                    
    print(f"Saved perfect CSV to {output_path}")

def load_csv_keypoints(csv_path):
    data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    
    if len(data) == 0:
        return np.array([])
        
    frames = data[:, 0].astype(int)
    joints = data[:, 1].astype(int)
    
    num_frames = np.max(frames) + 1
    actual_num_joints = 13 # MoveNet-13
    
    keypoints = np.zeros((num_frames, actual_num_joints, 2))
    for i in range(len(data)):
        f = int(data[i, 0])
        j = int(data[i, 1])
        if j < actual_num_joints:
            keypoints[f, j, 0] = data[i, 2]
            keypoints[f, j, 1] = data[i, 3] 
            
    return keypoints

def load_gt_keypoints(gt_path, subject, action, camera_idx):
    if not os.path.exists(gt_path):
        print(f"Error: GT file not found: {gt_path}")
        return None
        
    data = np.load(gt_path, allow_pickle=True)
    positions_2d = data['positions_2d'].item()
    
    if subject not in positions_2d:
        print(f"Error: Subject {subject} not found in GT.")
        return None
        
    target_action = None
    for a in positions_2d[subject].keys():
        if a.startswith(action):
            target_action = a
            break
            
    if target_action is None:
        print(f"Error: Action {action} not found for {subject} in GT.")
        return None
        
    seqs = positions_2d[subject][target_action]
    if camera_idx >= len(seqs):
        print(f"Error: Camera index {camera_idx} out of range for GT sequence.")
        return None
        
    return seqs[camera_idx]

def visualize_comparison(csv_kps, gt_kps, frame_idx, save_path=None, perfect=False):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    dhp19_connections = [
        (0, 10), (10, 12), (6, 9), (9, 11), (3, 5), (5, 8), (2, 4), (4, 7),
        (0, 6), (3, 2), (0, 3), (6, 2), (3, 1), (2, 1)
    ]
    dhp19_labels = ['hipL', 'head', 'sR', 'sL', 'eR', 'eL', 'hipR', 'wR', 'wL', 'kR', 'kL', 'aR', 'aL']

    if perfect:
        csv_connections = dhp19_connections
        csv_labels = dhp19_labels
        csv_title = f"Perfect CSV (DHP19) - Frame {frame_idx}"
    else:
        # MoveNet connections (index 0-12)
        csv_connections = [
            (1, 2), (1, 0), (2, 0), (4, 8), (1, 3), (3, 7), (2, 4), (4, 8), 
            (1, 6), (6, 5), (5, 2), (6, 9), (9, 11), (5, 10), (10, 12), (1, 0), (2, 0)
        ]
        csv_labels = ['head', 'sR', 'sL', 'eR', 'eL', 'hL', 'hR', 'wR', 'wL', 'kR', 'kL', 'aR', 'aL']
        csv_title = f"CSV (MoveEnet) - Frame {frame_idx}"

    def plot_pose(ax, pose, connections, labels, title, flip_x=False, flip_y=False, draw_edges=True):
        plot_pose_data = pose.copy()
        if flip_x:
            plot_pose_data[:, 0] = 346.0 - plot_pose_data[:, 0]
        if flip_y:
            plot_pose_data[:, 1] = 260.0 - plot_pose_data[:, 1]
            
        if draw_edges:
            for i, j in connections:
                ax.plot([plot_pose_data[i, 0], plot_pose_data[j, 0]], 
                        [plot_pose_data[i, 1], plot_pose_data[j, 1]], color='blue', linewidth=2)
                        
        ax.scatter(plot_pose_data[:, 0], plot_pose_data[:, 1], color='red', s=40, zorder=5)
        for i, (x, y) in enumerate(plot_pose_data):
            if i < len(labels):
                ax.annotate(labels[i], (x, y), textcoords="offset points", xytext=(5,5), ha='center', fontsize=7)
        ax.set_title(title)
        # ax.invert_yaxis() # Keep disabled, we handle Y-up in data
        ax.set_aspect('equal')

    if csv_kps is not None and frame_idx < len(csv_kps):
        # If perfect, CSV is already Y-up (no flip). If not, flip MoveEnet to Y-up.
        plot_pose(ax1, csv_kps[frame_idx], csv_connections, csv_labels, csv_title, flip_y=(not perfect))
    else:
        ax1.set_title("CSV Keypoints Not Available")

    if gt_kps is not None and frame_idx < len(gt_kps):
        # GT is already Y-up, so no flip needed
        plot_pose(ax2, gt_kps[frame_idx], dhp19_connections, dhp19_labels, f"Ground Truth (DHP19) - Frame {frame_idx}", flip_x=False, flip_y=False)
    else:
        ax2.set_title("GT Keypoints Not Available")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Visualize MoveEnet CSV Keypoints vs Ground Truth')
    parser.add_argument('--csv', type=str, required=True, help='Path to keypoints CSV file')
    parser.add_argument('--gt', type=str, help='Path to 2D GT NPZ file (e.g. data_2d_dhp19_gt.npz)')
    parser.add_argument('--subject', type=str, help='Subject (e.g. S13)')
    parser.add_argument('--action', type=str, help='Action index (e.g. 1)')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (0-3)')
    parser.add_argument('--frame', type=int, default=0, help='Frame index to visualize')
    parser.add_argument('--save', type=str, help='Output path to save the image')
    parser.add_argument('--save-csv', type=str, help='Output path to save the perfect CSV')
    parser.add_argument('--perfect', action='store_true', help='Input CSV is already in perfect DHP19 format')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv):
        print(f"Error: CSV file not found: {args.csv}")
        return
        
    print(f"Loading CSV keypoints from {args.csv}")
    csv_keypoints = load_csv_keypoints(args.csv)
    
    gt_keypoints = None
    if args.gt and args.subject and args.action:
        print(f"Loading GT keypoints from {args.gt} for {args.subject}_{args.action} cam {args.camera}")
        gt_keypoints = load_gt_keypoints(args.gt, args.subject, args.action, args.camera)
        
    if gt_keypoints is not None:
        print(f"Loaded {len(gt_keypoints)} GT frames")
        
        # Calculate MPJPE
        frame_mpjpe, overall_mpjpe = calculate_mpjpe(csv_keypoints, gt_keypoints, perfect=args.perfect)
        if frame_mpjpe is not None:
            print(f"Overall MPJPE: {overall_mpjpe:.2f} pixels")
            if len(frame_mpjpe) > 0:
                print(f"MPJPE Frame 0: {frame_mpjpe[0]:.2f} pixels")
            if len(frame_mpjpe) > 1:
                print(f"MPJPE Frame 1: {frame_mpjpe[1]:.2f} pixels")

    if args.save_csv:
        save_perfect_csv(csv_keypoints, args.save_csv)

    visualize_comparison(csv_keypoints, gt_keypoints, args.frame, args.save, perfect=args.perfect)
    
    # Print joints of the first 2 frames
    if args.perfect:
        print_labels = ['hipL', 'head', 'sR', 'sL', 'eR', 'eL', 'hipR', 'wR', 'wL', 'kR', 'kL', 'aR', 'aL']
        print_title = "[CSV Perfect (DHP19) Keypoints]"
    else:
        print_labels = ['head', 'sR', 'sL', 'eR', 'eL', 'hL', 'hR', 'wR', 'wL', 'kR', 'kL', 'aR', 'aL']
        print_title = "[CSV MoveEnet Keypoints]"
        
    dhp19_labels = ['hipL', 'head', 'sR', 'sL', 'eR', 'eL', 'hipR', 'wR', 'wL', 'kR', 'kL', 'aR', 'aL']
    
    print("\n" + "="*50)
    print("JOINT COORDINATES FOR FIRST 2 FRAMES")
    print("="*50)
    
    if csv_keypoints is not None and len(csv_keypoints) > 0:
        print(f"\n{print_title}")
        num_frames_to_print = min(2, len(csv_keypoints))
        for f in range(num_frames_to_print):
            print(f"  Frame {f}:")
            for j in range(len(print_labels)):
                if j < csv_keypoints.shape[1]:
                    x, y = csv_keypoints[f, j]
                    print(f"    Joint {j} ({print_labels[j]}): ({x:.2f}, {y:.2f})")
    
    if gt_keypoints is not None:
        print("\n[GT DHP19 Keypoints]")
        num_frames_to_print = min(2, len(gt_keypoints))
        for f in range(num_frames_to_print):
            print(f"  Frame {f}:")
            for j in range(len(dhp19_labels)):
                if j < gt_keypoints.shape[1]:
                    x, y = gt_keypoints[f, j]
                    # Note: visualization flips GT y, but here we print raw or flipped?
                    # User likely wants to see what's being compared. 
                    # Let's print raw but maybe mention if it's flipped in plot.
                    print(f"    Joint {j} ({dhp19_labels[j]}): ({x:.2f}, {y:.2f})")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
