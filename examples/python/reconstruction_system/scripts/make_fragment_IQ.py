
import sys
from pathlib import Path
import open3d as o3d
import numpy as np
from opencv_pose_estimation import pose_estimation

from os.path import join

pyexample_path = Path(__file__).parents[2].as_posix()
sys.path.append(pyexample_path)
# from open3d_example import *
from open3d_example import make_clean_folder, get_rgbd_file_lists, read_rgbd_image

init_config_path = Path(__file__).parent.as_posix()
sys.path.append(init_config_path)
from initialize_config import initialize_config
from optimize_posegraph import optimize_posegraph_for_fragment


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config):
    convert_rgb_to_intensity = config['convert_rgb_to_intensity']
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], convert_rgb_to_intensity, config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], convert_rgb_to_intensity, config)

    option = o3d.pipelines.odometry.OdometryOption()
    option.depth_diff_max = config["depth_diff_max"]
    if abs(s - t) != 1:
        if with_opencv:
            success_5pt, odo_init = pose_estimation(source_rgbd_image, target_rgbd_image, intrinsic, False)
            if success_5pt:
                [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(source_rgbd_image, target_rgbd_image, intrinsic, odo_init, o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
                return [success, trans, info]
        return [False, np.identity(4), np.identity(6)]
    else:
        odo_init = np.identity(4)
        [success, trans, info] = o3d.pipelines.odometry.compute_rgbd_odometry(
            source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files,
                                depth_files, fragment_id, n_fragments,
                                intrinsic, with_opencv, config,
                                trans_init=np.identity(4)):

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = trans_init
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):
            # odometry
            if t == s + 1:
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments - 1, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                                                 t - sid,
                                                                                 trans,
                                                                                 info,
                                                                                 uncertain=False))

            # keyframe loop closure
            if s % config['n_keyframes_per_n_frame'] == 0 and t % config['n_keyframes_per_n_frame'] == 0:
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments - 1, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid, t - sid, trans, info, uncertain=True))

    o3d.io.write_pose_graph(join(path_dataset, config["template_fragment_posegraph"] % fragment_id), pose_graph)

    pass


def make_posegraph_for_fragment_two_cameras(color_files, depth_files,
                                            sid_1, eid_1, sid_2, eid_2,
                                            fragment_id, n_fragments,
                                            intrinsic_1, intrinsic_2,
                                            config, with_opencv=True,
                                            trans_init_1=np.identity(4),
                                            trans_init_2=np.identity(4),
                                            ):

    path_dataset_1 = config['path_dataset']
    path_dataset_2 = config['path_dataset_2']

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pose_graph = o3d.pipelines.registration.PoseGraph()

    sid = sid_1
    eid = eid_2

    # first camera
    trans_odometry_1 = trans_init_1
    trans_odometry_2 = trans_init_2
    trans_odometry_2_inv = np.linalg.inv(trans_odometry_2)

    trans_odometry = trans_odometry_1

    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry))
    for s in range(sid, eid):
        for t in range(s + 1, eid):

            if s < sid_2 and t < sid_2:  # both images from first camera
                intrinsic = intrinsic_1
            elif s < sid_2 and t >= sid_2:  # s from first camera and t from second camera
                intrinsic = intrinsic_2  # FIXME: need 2 different intrinsics...
            elif s >= sid_2 and t < sid_2:  # s from second camera and t from first camera (maybe cannot be reached
                intrinsic = intrinsic_1  # FIXME: need 2 different intrinsics...
            elif s >= sid_2 and t >= sid_2:  # both images from second camera
                intrinsic = intrinsic_2

            # odometry
            if (t == s + 1) and (((s < sid_2) and (t < sid_2)) or ((s >= sid_2) and (t >= sid_2))):

                # if s < sid_2 and t < sid_2:  # both images from first camera
                #     trans_odometry = trans_odometry_1
                #     intrinsic = intrinsic_1
                # elif s < sid_2 and t >= sid_2:  # s from first camera and t from second camera
                #     trans_odometry = trans_odometry_2
                #     intrinsic = intrinsic_2  # FIXME: need 2 different intrinsics...
                # elif s >= sid_2 and t < sid_2:  # s from second camera and t from first camera (maybe cannot be reached
                #     trans_odometry = trans_odometry_2_inv
                #     intrinsic = intrinsic_1  # FIXME: need 2 different intrinsics...
                # elif s >= sid_2 and t >= sid_2:  # both images from second camera
                #     trans_odometry = np.identity(4)  # FIXME: not sure if needed identity or trans_odometry_2
                #     intrinsic = intrinsic_2

                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments - 1, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config)
                trans_odometry = np.dot(trans, trans_odometry)
                trans_odometry_inv = np.linalg.inv(trans_odometry)
                pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(trans_odometry_inv))
                pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid,
                                                                                 t - sid,
                                                                                 trans,
                                                                                 info,
                                                                                 uncertain=False))

                # save updated transformation
                # # FIXME: not sure if needed
                # if s < sid_2 and t < sid_2:  # both images from first camera
                #     trans_odometry_1 = trans_odometry
                # elif s < sid_2 and t >= sid_2:  # s from first camera and t from second camera
                #     trans_odometry_2 = trans_odometry
                # elif s >= sid_2 and t < sid_2:  # s from second camera and t from first camera (maybe cannot be reached
                #     trans_odometry_2_inv = trans_odometry
                # elif s >= sid_2 and t >= sid_2:  # both images from second camera
                #     trans_odometry_2 = trans_odometry

            # keyframe loop closure
            if (s % config['n_keyframes_per_n_frame'] == 0 and t % config['n_keyframes_per_n_frame'] == 0)\
                    or (((s < sid_2) and (t >= sid_2)) or ((s >= sid_2) and (t < sid_2))):
                print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d" % (fragment_id, n_fragments - 1, s, t))
                [success, trans, info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic, with_opencv, config)
                if success:
                    pose_graph.edges.append(o3d.pipelines.registration.PoseGraphEdge(s - sid, t - sid, trans, info, uncertain=True))

    pose_graph_file = join(path_dataset_1, config["template_fragment_posegraph"] % fragment_id)
    Path(pose_graph_file).parent.mkdir(exist_ok=True, parents=True)
    o3d.io.write_pose_graph(pose_graph_file, pose_graph)

    pass


def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config, intrinsic_2=None, sid_2=None):
    pose_graph = o3d.io.read_pose_graph(pose_graph_name)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=config["tsdf_cubic_size"] / 512.0,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for i in range(len(pose_graph.nodes)):
        i_abs = fragment_id * config['n_frames_per_fragment'] + i
        print(
            "Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)." %
            (fragment_id, n_fragments - 1, i_abs, i + 1, len(pose_graph.nodes)))
        rgbd = read_rgbd_image(color_files[i_abs], depth_files[i_abs], False,
                               config)
        pose = pose_graph.nodes[i].pose

        if (sid_2 is not None) and (intrinsic_2 is not None):
            intrinsic = intrinsic if i < sid_2 else intrinsic_2

        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config,
                                 intrinsic_2=None, sid_2=None):
    mesh = integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config, intrinsic_2, sid_2)
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd_name = join(path_dataset, config["template_fragment_pointcloud"] % fragment_id)
    o3d.io.write_point_cloud(pcd_name,
                             pcd,
                             # format='auto',
                             write_ascii=False,
                             compressed=True)
    pass


def get_config(path_dataset_1,
               path_intrinsic_1,
               path_dataset_2='',
               path_intrinsic_2='',
               n_keyframes_per_n_frame=1,
               depth_max=3.,
               depth_scale=3999.999810010204,
               python_multi_threading=False,
               output_root_dir='fragments',
               template_fragment_posegraph='fragment_%03d.json',
               template_fragment_posegraph_optimized='fragment_optimized_%03d.json',
               template_fragment_pointcloud='fragment_%03d.ply',
               debug_mode=False,
               icp_method='point_to_point',  # one of ['point_to_point', 'point_to_plane', 'color', 'generalized']
               ):

    print('Loading RealSense L515 Custom Dataset')

    # Get the dataset.
    # jackjack_bag = o3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = path_dataset_1
    config['path_intrinsic'] = path_intrinsic_1
    config['path_dataset_2'] = path_dataset_2
    config['path_intrinsic_2'] = path_intrinsic_2
    config['depth_max'] = depth_max
    config['depth_scale'] = depth_scale
    config['voxel_size'] = 0.01
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = python_multi_threading
    config['n_keyframes_per_n_frame'] = n_keyframes_per_n_frame
    config['n_frames_per_fragment'] = 75
    config['convert_rgb_to_intensity'] = True
    config['template_fragment_posegraph'] = (Path(output_root_dir) / template_fragment_posegraph).as_posix()
    config['template_fragment_posegraph_optimized'] = (Path(output_root_dir) / template_fragment_posegraph_optimized).as_posix()
    config['template_fragment_pointcloud'] = (Path(output_root_dir) / template_fragment_pointcloud).as_posix()
    config['debug_mode'] = debug_mode
    config['icp_method'] = icp_method

    # set all other config parameters
    initialize_config(config)

    return config


def main_single_camera():

    """
    adapted from: make_fragments.py::run()
    """

    # left camera 00
    path_dataset = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/'
    path_intrinsic = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/intrinsic_00_left.json'
    # right camera 01
    # path_dataset = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/'
    # path_intrinsic = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/intrinsic_01_right.json'

    depth_scale = 1 / 0.0002500000118743628

    config = get_config(path_dataset, path_intrinsic, depth_scale=depth_scale)

    print("making fragments from RGBD sequence.")
    make_clean_folder(Path(config["path_dataset"]) / config["folder_fragment"])

    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
    n_files = len(color_files)

    """
    adapted from: make_fragments.py::process_single_fragment() 
    """

    intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])

    fragment_id = 0
    n_fragments = 1
    n_max_images = 2  # -1

    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)
    if n_max_images > 0:
        eid = min(eid, n_max_images)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files, depth_files, fragment_id, n_fragments, intrinsic, True, config)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files, depth_files, fragment_id, n_fragments, intrinsic, config)

    pass

def main_two_cameras():

    # left camera 00 f0350845
    path_dataset_1 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/'
    path_intrinsic_1 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/intrinsic_00_left.json'
    # right camera 01 f1150179
    path_dataset_2 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/'
    path_intrinsic_2 = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/intrinsic_01_right.json'

    depth_scale = 1 / 0.0002500000118743628

    output_root_dir = 'fragments_two_cameras'
    config = get_config(path_dataset_1, path_intrinsic_1, path_dataset_2, path_intrinsic_2, depth_scale=depth_scale, output_root_dir=output_root_dir)


    # direct transformation
    # T = np.array([[0.618462, -0.428643,  0.658612,  -856.314],
    #               [0.428104,  0.886618,  0.175031,  -236.403],
    #               [-0.658963, 0.173705,  0.731843,   350.537],
    #               [-0,        -0,        -0,         1],
    #               ])

    # inverse transformation
    # T = np.array([[0.618462,  0.428104, -0.658963,   861.793],
    #               [-0.428643, 0.886618,  0.173705,  -218.344],
    #               [0.658612,  0.175031,  0.731843,   348.819],
    #               [-0,        -0,        -0,         1],
    #               ])

    # direct transformation saved in gitlab between right and left cameras of IQ 2.3 (f1150179-f0350845)
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957],
                  [0.45228962,    0.87499056,    0.17270096, -242.94946348],
                  [-0.64597401,   0.18787587,    0.73987852,  344.81816623],
                  [-0,        -0,        -0,         1],
                  ])
    T = np.linalg.inv(T)  # use inverse transformation


    print("making fragments from RGBD sequence.")
    # make_clean_folder(Path(config["path_dataset"]) / config["folder_fragment"])
    make_clean_folder(Path(config["path_dataset"]) / output_root_dir)

    [color_files_1, depth_files_1] = get_rgbd_file_lists(path_dataset_1)
    [color_files_2, depth_files_2] = get_rgbd_file_lists(path_dataset_2)
    # n_files = len(color_files)

    """
    adapted from: make_fragments.py::process_single_fragment() 
    """

    intrinsic_1 = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    intrinsic_2 = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic_2"])

    fragment_id = 0
    n_fragments = 1
    n_max_images = 2  # -1

    # sid = fragment_id * config['n_frames_per_fragment']
    # eid = min(sid + config['n_frames_per_fragment'], n_files)
    # if n_max_images > 0:
    #     eid = min(eid, n_max_images)

    sid_1 = 0
    eid_1 = sid_1 + n_max_images
    sid_2 = eid_1
    eid_2 = sid_2 + n_max_images

    color_files = color_files_1[sid_1:eid_1] + color_files_2[sid_1:eid_1]
    depth_files = depth_files_1[sid_1:eid_1] + depth_files_2[sid_1:eid_1]

    make_posegraph_for_fragment_two_cameras(color_files, depth_files,
                                            sid_1, eid_1, sid_2, eid_2,
                                            fragment_id, n_fragments,
                                            intrinsic_1, intrinsic_2,
                                            config, with_opencv=True,
                                            trans_init_1=np.identity(4),
                                            trans_init_2=T)
    optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    make_pointcloud_for_fragment(config["path_dataset"], color_files, depth_files, fragment_id, n_fragments, intrinsic_1, config, intrinsic_2, sid_2)

    pass


if __name__ == '__main__':

    # main_single_camera()
    main_two_cameras()

    pass