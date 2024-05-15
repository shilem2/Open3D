# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/integrate_scene.py

import numpy as np
import math
import os, sys
import open3d as o3d
from pathlib import Path

pyexample_path = Path(__file__).parents[2].as_posix()
sys.path.append(pyexample_path)
from open3d_example import get_rgbd_file_lists, join, read_rgbd_image, write_poses_to_log

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from make_fragment_IQ import get_config


def scalable_integrate_rgb_frames(path_dataset, intrinsic_1, intrinsic_2, config):

    poses = []
    [color_files_1, depth_files_1] = get_rgbd_file_lists(config['path_dataset'])
    [color_files_2, depth_files_2] = get_rgbd_file_lists(config['path_dataset_2'])

    n_max_images = 2  # -1
    n_fragments = 2

    color_files_1 = color_files_1[:n_max_images]
    depth_files_1 = depth_files_1[:n_max_images]
    color_files_2 = color_files_2[:n_max_images]
    depth_files_2 = depth_files_2[:n_max_images]

    # [color_files, depth_files] = get_rgbd_file_lists(path_dataset)
    # n_files = len(color_files)

    # n_fragments = int(math.ceil(float(n_files) / config['n_frames_per_fragment']))
    # TODO: optimize parameters
    volume = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=config["tsdf_cubic_size"] / 512.0, sdf_trunc=0.04, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    pose_graph_fragment = o3d.io.read_pose_graph(join(path_dataset, config["template_refined_posegraph_optimized"]))

    for fragment_id in range(len(pose_graph_fragment.nodes)):

        # FIXME: hardcoded, fix in in the future
        fragment_id_file = 0
        if fragment_id == 0:
            path_dataset_pose_graph = config['path_dataset']
            color_files = color_files_1
            depth_files = depth_files_1
            intrinsic = intrinsic_1
        elif fragment_id == 1:
            path_dataset_pose_graph = config['path_dataset_2']
            color_files = color_files_2
            depth_files = depth_files_2
            intrinsic = intrinsic_2

        pose_graph_rgbd = o3d.io.read_pose_graph(join(path_dataset_pose_graph, config["template_fragment_posegraph_optimized"] % fragment_id_file))

        for frame_id in range(len(pose_graph_rgbd.nodes)):

            frame_id_abs = fragment_id_file * config['n_frames_per_fragment'] + frame_id
            print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." % (fragment_id, n_fragments - 1, frame_id_abs, frame_id + 1, len(pose_graph_rgbd.nodes)))

            rgbd = read_rgbd_image(color_files[frame_id_abs], depth_files[frame_id_abs], False, config)
            pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose, pose_graph_rgbd.nodes[frame_id].pose)
            volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
            poses.append(pose)

            pass

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    if config["debug_mode"]:
        o3d.visualization.draw_geometries([mesh])

    mesh_name = join(path_dataset, config["template_global_mesh"])
    o3d.io.write_triangle_mesh(mesh_name, mesh, False, True)

    traj_name = join(path_dataset, config["template_global_traj"])
    write_poses_to_log(traj_name, poses)

    pass


# def run(config):
#
#     print("integrate the whole RGBD sequence using estimated camera pose.")
#
#     if config["path_intrinsic"]:
#         intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
#     else:
#         intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
#
#     scalable_integrate_rgb_frames(config["path_dataset"], intrinsic, config)


def main():

    # left camera 00 f0350845
    path_dataset_1 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/'
    path_intrinsic_1 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/intrinsic_00_left.json'
    # right camera 01 f1150179
    path_dataset_2 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/'
    path_intrinsic_2 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/intrinsic_01_right.json'

    depth_scale = 1 / 0.0002500000118743628

    output_root_dir = 'fragments_single_camera'

    icp_method = 'color'  # one of ['point_to_point', 'point_to_plane', 'color', 'generalized']

    template_fragment_posegraph_optimized = ''

    config = get_config(path_dataset_1, path_intrinsic_1, path_dataset_2, path_intrinsic_2, depth_scale=depth_scale,
                        output_root_dir=output_root_dir, icp_method=icp_method)

    intrinsic_1 = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    intrinsic_2 = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic_2"])

    scalable_integrate_rgb_frames(config["path_dataset"], intrinsic_1, intrinsic_2, config)

    pass


if __name__ == "__main__":

    main()

    pass