# -*- coding: utf-8 -*-
import os
import sys
import json
import argparse
import numpy as np
import scipy.linalg as la
import random
import glob
import joblib
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def calc_smallest_non_trivial_eigenvectors(bone_names, adjacency_dict):
    """ Functions """
    def create_adjacency_matrix(bone_names, adjacency_dict):
        n = len(bone_names)
        A = np.zeros((n, n))  # Initialize the adjacency matrix
        for bone, neighbors in adjacency_dict.items():
            bone_index = bone_names.index(bone)
            for neighbor in neighbors:
                neighbor_index = bone_names.index(neighbor)
                A[bone_index, neighbor_index] = 1
                A[neighbor_index, bone_index] = 1  # Ensure the matrix is symmetric
        return A

    def cosine_similarity(v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    def calculate_similarity_for_k(k, eigenvectors):
        selected_vectors = eigenvectors[:, :k]  # 最初のk個の非自明固有ベクトルを取得
        num_nodes = selected_vectors.shape[0]
        similarity_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                similarity_matrix[i, j] = cosine_similarity(selected_vectors[i], selected_vectors[j])
                
        return similarity_matrix

    def matrix_difference_norm(matrix1, matrix2):
        return np.linalg.norm(matrix1 - matrix2)

    """ Calculation of non-trivial eigenvectors """
    A = create_adjacency_matrix(bone_names, adjacency_dict)
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigenvalues, eigenvectors = la.eigh(L)
    non_trivial_indices = np.where(eigenvalues > 1e-10)[0]  
    non_trivial_eigenvalues = eigenvalues[non_trivial_indices]
    non_trivial_eigenvectors = eigenvectors[:, non_trivial_indices]

    """ 
    Calculate the norm of the difference between the similarity matrix 
    and the adjacency matrix A for values from 1 to 50.
    """
    norm_differences = []
    max_k = 50
    for k in range(1, max_k + 1):
        if k <= non_trivial_eigenvectors.shape[1]:
            similarity_matrix_k = calculate_similarity_for_k(k, non_trivial_eigenvectors)
            norm_diff = matrix_difference_norm(similarity_matrix_k, A)
            norm_differences.append(norm_diff)
        else:
            break  

    """ Find the value of K with the smallest difference. """
    min_norm_diff = min(norm_differences)
    best_k = norm_differences.index(min_norm_diff) + 1  
    smallest_non_trivial_eigenvectors = non_trivial_eigenvectors[:, :best_k]  

    return smallest_non_trivial_eigenvectors


def generate_smooth_motion_data(B, T, smoothness=10):
    ranges = [
        (-10, 10),  # X parameters in range -10 to 10
        (0, 2),     # Y parameter in range 0 to 2
        (-10, 10),  # Z parameters in range -10 to 10
        (-1, 1),    # sin(roll) parameters in range -1 to 1
        (-1, 1),    # cos(roll) parameters in range -1 to 1
        (-1, 1),    # sin(pitch) parameters in range -1 to 1
        (-1, 1),    # cos(pitch) parameters in range -1 to 1
        (-1, 1),    # sin(yaw) parameters in range -1 to 1
        (-1, 1)     # cos(yaw) parameters in range -1 to 1
    ]
    num_dims = len(ranges)
    # Initialize the motion data array
    motion_data = np.zeros((B, T, num_dims))
    for b in range(B):
        for d in range(num_dims):
            # Generate control points for smoothness
            control_points = np.random.rand(smoothness) * (ranges[d][1] - ranges[d][0]) + ranges[d][0]
            control_times = np.linspace(0, T - 1, smoothness)
            # Interpolate with a smooth function
            interpolator = interp1d(control_times, control_points, kind="cubic")
            # Generate the smooth time-series data for this dimension
            time_series = interpolator(np.arange(T))
            # Assign the generated data to the array
            motion_data[b, :, d] = time_series
    return motion_data


def normalize_coordinates(tensor):
    # Set the mean position of the x and z coordinates in the tensor data to zero.
    mean_x = np.mean(tensor[:, :, 0])
    mean_z = np.mean(tensor[:, :, 2])
    tensor[:, :, 0] -= mean_x
    tensor[:, :, 2] -= mean_z
    return tensor

def convert_to_euler_angles(data):
    frames, bones, _ = data.shape
    euler_angles = np.zeros((frames, bones, 3)) 
    for i in range(frames):
        for j in range(bones):
            cos_x, sin_x, cos_y, sin_y, cos_z, sin_z = data[i, j]
            x_angle = np.arctan2(sin_x, cos_x)
            y_angle = np.arctan2(sin_y, cos_y)
            z_angle = np.arctan2(sin_z, cos_z)
            euler_angles[i, j] = [x_angle, y_angle, z_angle]
    return euler_angles


def calculate_scalar_velocity_acceleration(positions):
    T = positions.shape[0]
    distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=2))
    velocities = np.zeros((T, positions.shape[1]))
    velocities[1:-1] = (distances[:-1] + distances[1:]) / 2
    velocities[0] = distances[0]
    velocities[-1] = distances[-1]
    accelerations = np.zeros((T, positions.shape[1]))
    accelerations[1:-1] = (velocities[2:] - velocities[:-2]) / 2
    accelerations[0] = velocities[1] - velocities[0]
    accelerations[-1] = velocities[-1] - velocities[-2] 
    return velocities, accelerations


def triangle_area_series(positions, bone_indices):
    if len(bone_indices) != 3:
        raise ValueError("Exactly three bone indices must be provided.")
    p1 = positions[:, bone_indices[0], :]
    p2 = positions[:, bone_indices[1], :]
    p3 = positions[:, bone_indices[2], :]
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = np.cross(v1, v2)
    area = np.linalg.norm(cross_product, axis=1) / 2
    return area

def calculate_volume_series(positions):
    min_vals = np.min(positions, axis=1)  # shape: (T, 3)
    max_vals = np.max(positions, axis=1)  # shape: (T, 3)
    dimensions = max_vals - min_vals  # shape: (T, 3)
    volumes = dimensions[:, 0] * dimensions[:, 1] * dimensions[:, 2]
    return volumes

def calculate_bone_distance_series(positions, bone_indices):
    if len(bone_indices) != 2:
        raise ValueError("Exactly two bone indices must be provided.")
    bone1_positions = positions[:, bone_indices[0], :]
    bone2_positions = positions[:, bone_indices[1], :]
    distances = np.linalg.norm(bone1_positions - bone2_positions, axis=1)
    return distances

def calculate_angle_series(positions, bone_indices):
    if len(bone_indices) != 3:
        raise ValueError("Exactly three bone indices must be provided.")
    boneA_positions = positions[:, bone_indices[0], :]
    boneB_positions = positions[:, bone_indices[1], :]
    boneC_positions = positions[:, bone_indices[2], :]
    vector_BA = boneA_positions - boneB_positions
    vector_BC = boneC_positions - boneB_positions
    norm_BA = np.linalg.norm(vector_BA, axis=1)
    norm_BC = np.linalg.norm(vector_BC, axis=1)
    dot_product = np.einsum('ij,ij->i', vector_BA, vector_BC)
    cos_theta = dot_product / (norm_BA * norm_BC)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angles = np.arccos(cos_theta)/np.pi
    return angles


def main(args):

    bone_names = json.loads(open(args.bone_name).read())
    bone2idx = dict([(bone_name, idx) for idx, bone_name in enumerate(bone_names)])
    bone_size = len(bone_names)
    adjacency_dict = json.loads(open(args.adjacency_dict).read())

    """ Calculation of Laplacian Positional Embedding (LPE) """
    lpe = calc_smallest_non_trivial_eigenvectors(bone_names, adjacency_dict)
    joblib.dump(lpe, "./lpe.joblib", compress=3)


    """ Generate pseudo motion """
    id2feature = {}
    for i, lyrics_file in enumerate(sorted(glob.glob(args.dir_lyrics.rstrip("/") + "/*.joblib"))):
        sys.stderr.write("\rGenerating pseudo motion [%03d/%03d]"%(i+1, 100))
        lyrics_dat = joblib.load(lyrics_file)
        song_id = lyrics_file.split("/")[-1].split(".")[0]
        n_frames = lyrics_dat[-1][-1]["frame"]+1
        motion_data = generate_smooth_motion_data(B=bone_size, T=n_frames)  # (Bone, T, Parameters)
        motion_data = np.transpose(motion_data, axes=(1, 0, 2))             # (T, Bone, Parameters)
        positions = motion_data[:, :, :3]
        angles = motion_data[:, :, 3:]
        euler_angles = convert_to_euler_angles(angles)

        """ Calculate affective features"""
        affective_feature = []
        # Area
        affective_feature.append(triangle_area_series(positions, [bone2idx["Head"], bone2idx["RightWrist"], bone2idx["LeftWrist"]]))
        affective_feature.append(triangle_area_series(positions, [bone2idx["Head"], bone2idx["LowerBody"], bone2idx["RightWrist"]]))
        affective_feature.append(triangle_area_series(positions, [bone2idx["Head"], bone2idx["LowerBody"], bone2idx["LeftWrist"]]))
        affective_feature.append(triangle_area_series(positions, [bone2idx["LowerBody"], bone2idx["RightWrist"], bone2idx["LeftWrist"]]))
        affective_feature.append(triangle_area_series(positions, [bone2idx["LowerBody"], bone2idx["RightAnkle"], bone2idx["LeftAnkle"]]))
        # Volume
        affective_feature.append(calculate_volume_series(positions))
        # Distance
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["Head"], bone2idx["RightWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["Head"], bone2idx["LeftWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["Head"], bone2idx["LowerBody"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["RightWrist"], bone2idx["LeftWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["LowerBody"], bone2idx["RightWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["LowerBody"], bone2idx["LeftWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["RightShoulder"], bone2idx["RightWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["LeftShoulder"], bone2idx["LeftWrist"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["LowerBody"], bone2idx["RightElbow"]]))
        affective_feature.append(calculate_bone_distance_series(positions, [bone2idx["LowerBody"], bone2idx["LeftElbow"]]))
        # Finger Curvature
        affective_feature.append(calculate_angle_series(positions, [bone2idx["RightThumb0"], bone2idx["RightThumb1"], bone2idx["RightThumb2"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["RightIndex1"], bone2idx["RightIndex2"], bone2idx["RightIndex3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["RightMiddle1"], bone2idx["RightMiddle2"], bone2idx["RightMiddle3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["RightRing1"], bone2idx["RightRing2"], bone2idx["RightRing3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["RightPinky1"], bone2idx["RightPinky2"], bone2idx["RightPinky3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftThumb0"], bone2idx["LeftThumb1"], bone2idx["LeftThumb2"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftIndex1"], bone2idx["LeftIndex2"], bone2idx["LeftIndex3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftMiddle1"], bone2idx["LeftMiddle2"], bone2idx["LeftMiddle3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftRing1"], bone2idx["LeftRing2"], bone2idx["LeftRing3"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftPinky1"], bone2idx["LeftPinky2"], bone2idx["LeftPinky3"]]))
        # Body Curvature
        affective_feature.append(calculate_angle_series(positions, [bone2idx["RightShoulder"], bone2idx["RightElbow"], bone2idx["RightWrist"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftShoulder"], bone2idx["LeftElbow"], bone2idx["LeftWrist"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Head"], bone2idx["UpperBody"], bone2idx["LowerBody"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Head"], bone2idx["Neck"], bone2idx["RightElbow"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Head"], bone2idx["Neck"], bone2idx["LeftElbow"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Neck"], bone2idx["RightElbow"], bone2idx["RightWrist"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Neck"], bone2idx["LeftElbow"], bone2idx["LeftWrist"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Neck"], bone2idx["RightWrist"], bone2idx["LowerBody"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["Neck"], bone2idx["LeftWrist"], bone2idx["LowerBody"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LowerBody"], bone2idx["RightKnee"], bone2idx["RightAnkle"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LowerBody"], bone2idx["LeftKnee"], bone2idx["LeftAnkle"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["LeftKnee"], bone2idx["LowerBody"], bone2idx["RightKnee"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["UpperBody"], bone2idx["LowerBody"], bone2idx["RightKnee"]]))
        affective_feature.append(calculate_angle_series(positions, [bone2idx["UpperBody"], bone2idx["LowerBody"], bone2idx["RightKnee"]]))

        affective_feature = np.column_stack(affective_feature)
        affective_size = affective_feature.shape[1]


        """ Calculate velocity/acceleration """
        point_velocities, point_accelerations = calculate_scalar_velocity_acceleration(positions)
        point_velocities = point_velocities[:, :, np.newaxis]
        point_accelerations = point_accelerations[:, :, np.newaxis]

        angle_vels = np.diff(angles, axis=0, prepend=np.zeros((1, bone_size, 6)))
        angle_accs = np.diff(angle_vels, axis=0, prepend=np.zeros((1, bone_size, 6)))

        body_directions = euler_angles[:, bone2idx["UpperBody"], :]
        rotations = R.from_euler('xyz', body_directions, degrees=False)
        local_positions = np.array([rot.apply(pos) for rot, pos in zip(rotations, positions)])
        local_velocities = np.diff(local_positions, axis=0, prepend=np.zeros((1, bone_size, 3)))
        local_accelerations = np.diff(local_velocities, axis=0, prepend=np.zeros((1, bone_size, 3)))

        affective_velocities = np.diff(affective_feature, axis=0, prepend=np.zeros((1, affective_size)))
        affective_accelerations = np.diff(affective_velocities, axis=0, prepend=np.zeros((1, affective_size)))

        """ Divide by bar """
        os.makedirs("./motion_data", exist_ok=True)
        output_motions = []
        for j, bar_lyrics in enumerate(lyrics_dat):
            n_bar_frames = len(bar_lyrics)
            bar_start = bar_lyrics[0]["frame"]
            bar_end = bar_start + n_bar_frames
            bar_angle = angles[bar_start:bar_end]       
            bar_position = positions[bar_start:bar_end]
            if len(bar_angle) == 0:
                continue
            # XZ coordinate adjustment (set the mean position to zero)
            bar_position = normalize_coordinates(bar_position)
            bar_point_velocity = point_velocities[bar_start:bar_end]
            bar_point_acceleration = point_accelerations[bar_start:bar_end]
            bar_local_velocity = local_velocities[bar_start:bar_end]
            bar_local_acceleration = local_accelerations[bar_start:bar_end]
            bar_angle_velocity = angle_vels[bar_start:bar_end]
            bar_angle_acceleration = angle_accs[bar_start:bar_end]
            bar_affective = affective_feature[bar_start:bar_end]
            bar_aft_velocity = affective_velocities[bar_start:bar_end]
            bar_aft_acceleration = affective_accelerations[bar_start:bar_end]
            if len(bar_point_velocity) != len(bar_point_acceleration):
                continue
            if len(bar_angle) == len(bar_position) == len(bar_point_velocity) == len(bar_point_acceleration) == len(bar_affective):
                bar_ske_feature = np.concatenate([bar_angle, bar_angle_velocity, bar_angle_acceleration, bar_position, bar_point_velocity, bar_point_acceleration, bar_local_velocity, bar_local_acceleration], axis=2)
                bar_aft_feature = np.concatenate([bar_affective, bar_aft_velocity, bar_aft_acceleration], axis=1)
                id2feature[(song_id, bar_start)] = {"skeletal_feature":bar_ske_feature.copy(), "affective_feature":bar_aft_feature.copy()}
    sys.stderr.write("\n")


    """ Normalize features """
    ske_features = np.concatenate([v["skeletal_feature"] for k, v in id2feature.items()], axis=0)
    ske_max = np.max(ske_features, axis=0)
    ske_min = np.min(ske_features, axis=0)
    ske_range = np.maximum(np.abs(ske_max), np.abs(ske_min))
    ske_range[ske_range == 0] = 1
    aft_features = np.concatenate([v["affective_feature"] for k, v in id2feature.items()], axis=0)
    aft_max = np.max(aft_features, axis=0)
    aft_min = np.min(aft_features, axis=0)
    output = defaultdict(list)
    for k, features in sorted(id2feature.items()):
        song_id = k[0]
        start_frame = k[1]
        """
        The skeletal feature is normalized between -1 and 1, 
        and the effective feature is normalized between 0 and 1.
        """
        norm_ske_feature = features["skeletal_feature"] / ske_range
        norm_aft_feature = (features["affective_feature"] - aft_min) / (aft_max - aft_min)
        norm_features = {"start_frame":start_frame, "skeletal_feature":norm_ske_feature, "affective_feature":norm_aft_feature}
        output[song_id].append(norm_features)

    """ Save data """
    for song_id, norm_features in sorted(output.items()):
        joblib.dump(norm_features, "./motion_data/%s.joblib"%song_id, compress=3)



    





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir_lyrics", "--dir_lyrics", dest="dir_lyrics", 
                        default="./lyrics_data", type=str, help="lyrics directory path")
    parser.add_argument("-bone_name", "--bone_name", dest="bone_name", 
                        default="./bone_names.json", type=str, help="json file path for bone names")
    parser.add_argument("-adjacency_dict", "--adjacency_dict", dest="adjacency_dict", 
                        default="./adjacency_dict.json", type=str, help="json file path for adjacency dict")
    args = parser.parse_args()
    main(args)

