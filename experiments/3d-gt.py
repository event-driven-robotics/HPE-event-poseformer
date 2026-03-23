import numpy as np
import scipy.io
from pathlib import Path
import argparse

JOINT_NAMES = [
    "head",
    "shoulderR",
    "shoulderL",
    "elbowR",
    "elbowL",
    "hipR",
    "hipL",
    "handR",
    "handL",
    "kneeL",
    "kneeR",
    "footL",
    "footR",
]

def convert_mat_to_flattened_predictions_format(mat_file_path, npz_file_path):
    mat_file_path = Path(mat_file_path)
    npz_file_path = Path(npz_file_path)

    # Load .mat
    mat_data = scipy.io.loadmat(mat_file_path)
    print(f"Keys in .mat file: {mat_data.keys()}")

    if "XYZPOS" not in mat_data:
        raise KeyError("Expected key 'XYZPOS' not found in .mat file")

    # XYZPOS is a tuple of 13 arrays, each corresponding to a joint
    xyzpos = mat_data["XYZPOS"]

    # Unwrap the tuple: this contains 13 joint arrays
    xyzpos = xyzpos[0, 0]

    # Check the structure of the data
    print(f"Shape of XYZPOS: {xyzpos.shape}")
    print(f"Content of XYZPOS: {xyzpos}")

    # Create a dictionary to store joint positions in the predictions format
    num_frames = xyzpos[0].shape[0]  # Number of frames
    flattened_predictions = np.zeros((num_frames, len(JOINT_NAMES), 3), dtype=np.float32)

    for j, joint_name in enumerate(JOINT_NAMES):
        joint_coords = xyzpos[j]  # (num_frames, 3) for each joint
        print(f"Joint: {joint_name} | Shape: {joint_coords.shape}")

        # Assign joint data to the correct position in flattened_predictions (frame, joint, xyz)
        flattened_predictions[:, j, :] = joint_coords
    # Reshape to (num_frames * num_joints, 3)
    flattened_predictions = flattened_predictions.reshape((num_frames * len(JOINT_NAMES), 3))

    # Save as NPZ (flattened predictions format)
    np.savez_compressed(npz_file_path, predictions=flattened_predictions)

    print(f"Converted GT saved in flattened prediction format → {npz_file_path}")
    print(f"Frames: {num_frames}, Joints: {len(JOINT_NAMES)}")
    print(f"Flattened Predictions Shape: {flattened_predictions.shape}")
    print(flattened_predictions[0:14])  # Print first frame (13 joints)

# Main function to handle command-line arguments
if __name__ == '__main__':
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert ground truth .mat file to predictions .npz format.")
    parser.add_argument('--input_path', type=str, help="Path to the input .mat file")
    parser.add_argument('--output_path', type=str, help="Path to save the output .npz file")

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided paths
    convert_mat_to_flattened_predictions_format(args.input_path, args.output_path)
