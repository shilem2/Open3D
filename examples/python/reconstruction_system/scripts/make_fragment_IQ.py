
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


def register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                           with_opencv, config):
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
                                intrinsic, with_opencv, config):

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    trans_odometry = np.identity(4)
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

def integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id,
                                      n_fragments, pose_graph_name, intrinsic,
                                      config):
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
        volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    return mesh

def make_pointcloud_for_fragment(path_dataset, color_files, depth_files,
                                 fragment_id, n_fragments, intrinsic, config):
    mesh = integrate_rgb_frames_for_fragment(color_files, depth_files, fragment_id, n_fragments,
        join(path_dataset,config["template_fragment_posegraph_optimized"] % fragment_id),
        intrinsic, config)
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
               path_intrinsic_2=None,
               n_keyframes_per_n_frame=1,
               depth_max=3.,
               depth_scale=3999.999810010204,
               python_multi_threading=False
               ):
    print('Loading RealSense L515 Custom Dataset')

    # Get the dataset.
    # jackjack_bag = o3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = path_dataset_1
    config['path_intrinsic'] = path_intrinsic_1
    config['depth_max'] = depth_max
    config['depth_scale'] = depth_scale
    config['voxel_size'] = 0.01
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = python_multi_threading
    config['n_keyframes_per_n_frame'] = n_keyframes_per_n_frame
    config['n_frames_per_fragment'] = 75
    config['convert_rgb_to_intensity'] = True

    # set all other config parameters
    initialize_config(config)

    return config


def main():

    """
    adapted from: make_fragments.py::run()
    """

    # left camera 00
    path_dataset = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/'
    path_intrinsic = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/intrinsic_00_left.json'
    # right camera 01
    path_dataset = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/'
    path_intrinsic = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/intrinsic_01_right.json'


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


if __name__ == '__main__':

    main()

    pass