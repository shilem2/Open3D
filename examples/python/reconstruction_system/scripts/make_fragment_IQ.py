
import sys
from pathlib import Path


pyexample_path = Path(__file__).parents[2].as_posix()
sys.path.append(pyexample_path)

# from open3d_example import *
from open3d_example import make_clean_folder

import open3d as o3d
import numpy as np


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



def main():


    """
    adapted from: make_fragments.py::run()
    """

    print("making fragments from RGBD sequence.")
    make_clean_folder(join(config["path_dataset"], config["folder_fragment"]))

    [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])


    """
    adapted from: make_fragments.py::process_single_fragment() 
    """

    if config["path_intrinsic"]:
        intrinsic = o3d.io.read_pinhole_camera_intrinsic(config["path_intrinsic"])
    else:
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    sid = fragment_id * config['n_frames_per_fragment']
    eid = min(sid + config['n_frames_per_fragment'], n_files)

    make_posegraph_for_fragment(config["path_dataset"], sid, eid, color_files, depth_files, fragment_id, n_fragments, intrinsic, with_opencv, config)
    # optimize_posegraph_for_fragment(config["path_dataset"], fragment_id, config)
    # make_pointcloud_for_fragment(config["path_dataset"], color_files, depth_files, fragment_id, n_fragments, intrinsic, config)





    pass


if __name__ == '__main__':

    main()

    pass