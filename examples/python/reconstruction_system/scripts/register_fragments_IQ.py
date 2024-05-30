import multiprocessing
import sys
import os
from pathlib import Path
import open3d as o3d
import numpy as np

from os.path import join

pyexample_path = Path(__file__).parents[2].as_posix()
sys.path.append(pyexample_path)
# from open3d_example import *
from open3d_example import make_clean_folder, get_rgbd_file_lists, read_rgbd_image, draw_registration_result

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from refine_registration_IQ import multiscale_icp
from make_fragment_IQ import get_config

init_config_path = Path(__file__).parent.as_posix()
sys.path.append(init_config_path)
from optimize_posegraph import optimize_posegraph_for_scene


def preprocess_point_cloud(pcd, config):

    voxel_size = config["voxel_size"]
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0, max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0, max_nn=100))

    return (pcd_down, pcd_fpfh)


def register_point_cloud_fpfh(source, target, source_fpfh, target_fpfh, config):

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    distance_threshold = config["voxel_size"] * 1.4

    if config["global_registration"] == "fgr":
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching( source, target, source_fpfh, target_fpfh,
                                                                                        o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold))

    if config["global_registration"] == "ransac":
        # Fallback to preset parameters that works better
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source, target, source_fpfh, target_fpfh, False, distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 4,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.999))

    if (result.transformation.trace() == 4.0):
        return (False, np.identity(4), np.zeros((6, 6)))

    information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, distance_threshold, result.transformation)

    if information[5, 5] / min(len(source.points), len(target.points)) < 0.3:
        return (False, np.identity(4), np.zeros((6, 6)))

    return (True, result.transformation, information)


def compute_initial_registration(s, t, source_down, target_down, source_fpfh, target_fpfh, path_dataset, config, trans_init=None):

    if t == s + 1:  # odometry case
        print("Using RGBD odometry")
        if trans_init is None:
            pose_graph_frag = o3d.io.read_pose_graph(join(path_dataset, config["template_fragment_posegraph_optimized"] % s))
            n_nodes = len(pose_graph_frag.nodes)
            transformation_init = np.linalg.inv(pose_graph_frag.nodes[n_nodes - 1].pose)
        else:
            transformation_init = trans_init
        (transformation, information) = multiscale_icp(source_down, target_down, [config["voxel_size"]], [50], config, transformation_init)

    else:  # loop closure case
        (success, transformation, information) = register_point_cloud_fpfh(source_down, target_down, source_fpfh, target_fpfh, config)

        if not success:
            print("No reasonable solution. Skip this pair")
            return (False, np.identity(4), np.zeros((6, 6)))

    print(transformation)

    if config["debug_mode"]:
        # draw_registration_result(source_down, target_down, trans_init)
        draw_registration_result(source_down, target_down, transformation)

    return (True, transformation, information)


def update_posegraph_for_scene(s, t, transformation, information, odometry,
                               pose_graph):

    if t == s + 1:  # odometry case
        odometry = np.dot(transformation, odometry)
        odometry_inv = np.linalg.inv(odometry)
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry_inv))
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, transformation, information, uncertain=False))

    else:  # loop closure case
        pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s, t, transformation, information, uncertain=True))

    return (odometry, pose_graph)


def register_point_cloud_pair(ply_file_names, s, t, config, trans_init=None):

    print("reading %s ..." % ply_file_names[s])
    source = o3d.io.read_point_cloud(ply_file_names[s])
    print("reading %s ..." % ply_file_names[t])
    target = o3d.io.read_point_cloud(ply_file_names[t])

    (source_down, source_fpfh) = preprocess_point_cloud(source, config)
    (target_down, target_fpfh) = preprocess_point_cloud(target, config)

    (success, transformation, information) = compute_initial_registration(s, t, source_down, target_down, source_fpfh, target_fpfh, config["path_dataset"], config, trans_init)

    if t != s + 1 and not success:
        return (False, np.identity(4), np.identity(6))

    if config["debug_mode"]:
        print(transformation)
        print(information)

    return (True, transformation, information)


# other types instead of class?
class matching_result:

    def __init__(self, s, t, trans_init=np.identity(4)):
        self.s = s
        self.t = t
        self.success = False
        self.transformation = trans_init
        self.infomation = np.identity(6)
        pass


def make_posegraph_for_scene(ply_file_names, config, trans_init=np.identity(4)):

    pose_graph = o3d.pipelines.registration.PoseGraph()
    # odometry = trans_init
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))

    n_files = len(ply_file_names)
    matching_results = {}
    for s in range(n_files):
        for t in range(s + 1, n_files):
            matching_results[s * n_files + t] = matching_result(s, t, trans_init)

    if config["python_multi_threading"] is True:
        os.environ['OMP_NUM_THREADS'] = '1'
        max_workers = max(1, min(multiprocessing.cpu_count() - 1, len(matching_results)))
        mp_context = multiprocessing.get_context('spawn')
        with mp_context.Pool(processes=max_workers) as pool:
            args = [(ply_file_names, v.s, v.t, config) for k, v in matching_results.items()]
            results = pool.starmap(register_point_cloud_pair, args)

        for i, r in enumerate(matching_results):
            matching_results[r].success = results[i][0]
            matching_results[r].transformation = results[i][1]
            matching_results[r].information = results[i][2]

    else:
        for r in matching_results:
            (matching_results[r].success, matching_results[r].transformation, matching_results[r].information) = \
                register_point_cloud_pair(ply_file_names, matching_results[r].s, matching_results[r].t, config, trans_init)

    for r in matching_results:
        if matching_results[r].success:
            (odometry, pose_graph) = update_posegraph_for_scene(matching_results[r].s, matching_results[r].t, matching_results[r].transformation, matching_results[r].information, odometry, pose_graph)
    o3d.io.write_pose_graph(join(config["path_dataset"], config["template_global_posegraph"]), pose_graph)


def main():

    print("register fragments.")

    # left camera 00 f0350845
    path_dataset_1 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/'
    path_intrinsic_1 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/intrinsic_00_left.json'
    # right camera 01 f1150179
    path_dataset_2 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/'
    path_intrinsic_2 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/intrinsic_01_right.json'

    depth_scale = 1 / 0.0002500000118743628

    output_root_dir = 'fragments_registration'

    icp_method = 'color'  # one of ['point_to_point', 'point_to_plane', 'color', 'generalized']

    config = get_config(path_dataset_1, path_intrinsic_1, path_dataset_2, path_intrinsic_2, depth_scale=depth_scale,
                        output_root_dir=output_root_dir, icp_method=icp_method)

    # direct transformation calculated by RecFusion calibration pattern
    # T = np.array([[0.618462, -0.428643,  0.658612,  -856.314/1000],
    #               [0.428104,  0.886618,  0.175031,  -236.403/1000],
    #               [-0.658963, 0.173705,  0.731843,   350.537/1000],
    #               [-0,        -0,        -0,         1],
    #               ])

    # direct transformation saved in gitlab
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957/1000],
                  [0.45228962,    0.87499056,    0.17270096, -242.94946348/1000],
                  [-0.64597401,   0.18787587,    0.73987852,  344.81816623/1000],
                  [-0,        -0,        -0,         1],
                  ])

    T = np.linalg.inv(T)  # use inverse transformation

    trans_init = T
    # trans_init = np.identity(4)

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # ply_file_names = get_file_list(join(config["path_dataset"], config["folder_fragment"]), ".ply")
    ply_file_names = ['/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/fragments_single_camera/fragment_000_processed.ply',
                      '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/fragments_single_camera/fragment_000_processed.ply'
                      ]

    make_clean_folder(join(config["path_dataset"], config["folder_scene"]))
    make_posegraph_for_scene(ply_file_names, config, trans_init)
    optimize_posegraph_for_scene(config["path_dataset"], config)

    pass


if __name__ == '__main__':

    main()

    pass