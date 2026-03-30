import numpy as np

# Load original 13-joint keypoints
data = np.load("input_2D/keypoints.npz")  # Replace with your actual file name
keypoints_13 = data["keypoints"]  # Assuming it's stored under the key 'keypoints'

# Check the shape: should be (N, 13, 2)
assert keypoints_13.shape[1] == 13, "Expected 13 joints"

# Map 13-joint indices
mapping_13 = {
    'head': 0, 'shoulder_right': 1, 'shoulder_left': 2, 'elbow_right': 3, 'elbow_left': 4,
    'hip_left': 5, 'hip_right': 6, 'wrist_right': 7, 'wrist_left': 8, 'knee_right': 9,
    'knee_left': 10, 'ankle_right': 11, 'ankle_left': 12
}

# Create new array for 17-joint keypoints
num_samples = keypoints_13.shape[0]
keypoints_17 = np.zeros((num_samples, 17, 2), dtype=np.float32)

# Fill directly mapped joints
keypoints_17[:, 0] = (keypoints_13[:, mapping_13['hip_left']] + keypoints_13[:, mapping_13['hip_right']]) / 2  # Hip (center)
keypoints_17[:, 1] = keypoints_13[:, mapping_13['hip_right']]  # RHip
keypoints_17[:, 2] = keypoints_13[:, mapping_13['knee_right']]  # RKnee
keypoints_17[:, 3] = keypoints_13[:, mapping_13['ankle_right']]  # RFoot
keypoints_17[:, 4] = keypoints_13[:, mapping_13['hip_left']]  # LHip
keypoints_17[:, 5] = keypoints_13[:, mapping_13['knee_left']]  # LKnee
keypoints_17[:, 6] = keypoints_13[:, mapping_13['ankle_left']]  # LFoot

# Calculate neck as average of both shoulders
neck = (keypoints_13[:, mapping_13['shoulder_left']] + keypoints_13[:, mapping_13['shoulder_right']]) / 2
keypoints_17[:, 8] = neck  # Neck

# Spine as average of neck and hip (center)
keypoints_17[:, 7] = (keypoints_17[:, 8] + keypoints_17[:, 0]) / 2  # Spine

# Nose as head in the 13-point dataset
nose = keypoints_13[:, mapping_13['head']]
head = 2 * nose - neck  # Since nose is the average of head and neck, head = 2 * nose - neck
keypoints_17[:, 9] = nose  # Nose (which is used as head in the 13-point dataset)
keypoints_17[:, 10] = head  # Head

# Upper limbs
keypoints_17[:, 11] = keypoints_13[:, mapping_13['shoulder_left']]  # LShoulder
keypoints_17[:, 12] = keypoints_13[:, mapping_13['elbow_left']]  # LElbow
keypoints_17[:, 13] = keypoints_13[:, mapping_13['wrist_left']]  # LWrist
keypoints_17[:, 14] = keypoints_13[:, mapping_13['shoulder_right']]  # RShoulder
keypoints_17[:, 15] = keypoints_13[:, mapping_13['elbow_right']]  # RElbow
keypoints_17[:, 16] = keypoints_13[:, mapping_13['wrist_right']]  # RWrist

# Save to a new .npz file
np.savez("input_2D/keypoints_17.npz", keypoints=np.array([keypoints_17]))
print("Saved converted keypoints to output_keypoints_17.npz")
