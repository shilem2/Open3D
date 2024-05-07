# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d


def lounge_data_loader():
    print('Loading Stanford Lounge RGB-D Dataset')

    # Get the dataset.
    lounge_rgbd = o3d.data.LoungeRGBDImages()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = lounge_rgbd.extract_dir
    config['path_intrinsic'] = ""
    config['depth_max'] = 3.0
    config['voxel_size'] = 0.05
    config['depth_diff_max'] = 0.07
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 3.0
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def bedroom_data_loader():
    print('Loading Redwood Bedroom RGB-D Dataset')

    # Get the dataset.
    bedroom_rgbd = o3d.data.BedroomRGBDImages()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = bedroom_rgbd.extract_dir
    config['path_intrinsic'] = ""
    config['depth_max'] = 3.0
    config['voxel_size'] = 0.05
    config['depth_diff_max'] = 0.07
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 3.0
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def jackjack_data_loader():
    print('Loading RealSense L515 Jack-Jack RGB-D Bag Dataset')

    # Get the dataset.
    jackjack_bag = o3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = jackjack_bag.path
    config['path_intrinsic'] = ""
    config['depth_max'] = 0.85
    config['voxel_size'] = 0.025
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def IQ_data_loader():
    print('Loading RealSense L515 IQ work volume Bag Dataset')

    # Get the dataset.
    # jackjack_bag = o3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec_sample/'
    config['path_intrinsic'] = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec_sample/intrinsic_00.json'
    config['depth_max'] = 3.
    config['voxel_size'] = 0.01
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = False
    config['n_keyframes_per_n_frame'] = 1

    return config

def realsense_data_loader():
    print('Loading RealSense L515 Custom Dataset')

    # Get the dataset.
    # jackjack_bag = o3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_150316_first_try/'
    config['path_intrinsic'] = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_150316_first_try/intrinsic.json'
    config['depth_max'] = 1.
    config['voxel_size'] = 0.01
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = False
    config['n_keyframes_per_n_frame'] = 3
    config['n_frames_per_fragment'] = 75

    return config
