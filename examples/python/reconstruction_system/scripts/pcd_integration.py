from pathlib import Path

import copy

import cv2
import numpy as np
import open3d as o3d
import cv2
import itertools
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

np.set_printoptions(precision=5, suppress=True)

import json
import time
import datetime
import os, sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

# from open3d_example import check_folder_structure

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from initialize_config import initialize_config, dataset_loader

import make_fragments
import register_fragments
import refine_registration
import integrate_scene
import slac
import slac_integrate

import matplotlib
matplotlib.use('MacOSX')


def pcd_integration():

    pcd0_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/00_points_0001.pcd'
    pcd1_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/01_points_0000.pcd'

    pcd0 = o3d.io.read_point_cloud(pcd0_file)
    pcd1 = o3d.io.read_point_cloud(pcd1_file)

    T_init = np.array([[1,   0,    0,   -2000],
                       [0,  -1,    0,   -2000],
                       [0,   0,   -1,   -1000],
                       [0,   0,    0,    1],
                       ])

    # T_init_0 = np.array([[0.99916,    -0.0279078, -0.0299502, -2099.5],
    #                      [-0.0283925, -0.999466,  -0.0159614, -1979.55],
    #                      [-0.0294902,  0.0167964, -0.999421,  -907.6],
    #                      [0,           0,          0,          1],
    #                      ])
    # T_init_1 = np.array([[0.999994,    0.00104767,  0.00389073, -1990.58],
    #                      [0.00104155, -0.999998,    0.00184154, -2001.92],
    #                      [0.00389326, -0.00183791, -0.999993,   -1000.65],
    #                      [0,           0,           0,           1],
    #                      ])

    # pcd0 = pcd0.transform(T_init)
    # pcd1 = pcd1.transform(T_init)
    # pcd0 = pcd0.transform(np.linalg.inv(T_init))
    # pcd1 = pcd1.transform(np.linalg.inv(T_init))
    # pcd0 = pcd0.transform(T_init_0)
    # pcd1 = pcd1.transform(T_init_1)
    # pcd0 = pcd0.transform(np.linalg.inv(T_init_0))
    # pcd1 = pcd1.transform(np.linalg.inv(T_init_1))


    # direct transformation
    # T = np.array([[0.618462, -0.428643,  0.658612,  -856.314],
    #               [0.428104,  0.886618,  0.175031,  -236.403],
    #               [-0.658963, 0.173705,  0.731843,   350.537],
    #               [-0,        -0,        -0,         1],
    #               ])

    # inverse transformation
    T = np.array([[0.618462,  0.428104, -0.658963,   861.793],
                  [-0.428643, 0.886618,  0.173705,  -218.344],
                  [0.658612,  0.175031,  0.731843,   348.819],
                  [-0,        -0,        -0,         1],
                  ])

    # direct transformation saved in gitlab
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    # T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957],
    #               [0.45228962,    0.87499056,    0.17270096, -242.94946348],
    #               [-0.64597401,   0.18787587,    0.73987852,  344.81816623],
    #               [-0,        -0,        -0,         1],
    #               ])
    # T = np.linalg.inv(T)  # use inverse transformation


    pcd0_T = copy.deepcopy(pcd0).transform(T)
    # pcd1_T = copy.deepcopy(pcd1).transform(T)
    # pcd1_T = copy.deepcopy(pcd1).transform(T)
    # pcd0_T = copy.deepcopy(pcd0).transform(T_init)
    # pcd1_T = copy.deepcopy(pcd1).transform(T_init)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd 0', width=1600, height=1400)

    # vis.add_geometry(pcd0)
    vis.add_geometry(pcd0_T)
    vis.add_geometry(pcd1)
    # vis.add_geometry(pcd1_T)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

    pass

def pcd_integration_IQ_meshes():

    pcd0_file = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/fragments_single_camera/fragment_000.ply'
    pcd1_file = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175527_IQ_right/fragments_single_camera/fragment_000.ply'

    display = False

    pcd0 = o3d.io.read_point_cloud(pcd0_file)
    pcd1 = o3d.io.read_point_cloud(pcd1_file)

    p = np.array(pcd0.points)
    print(p[:, 0].min(), p[:, 0].max())
    print(p[:, 1].min(), p[:, 1].max())
    print(p[:, 2].min(), p[:, 2].max())
    import pandas as pd
    df = pd.DataFrame({'x': p[:, 0], 'y': p[:, 1], 'z': p[:, 2]})
    print(df.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]).round(2))

    x_min_max = [-0.5, 0.5]
    y_min_max = [-0.5, 0.5]
    z_min_max = [1., 2.]

    pcd0_filtered = filter_pcd(pcd0, x_min_max, y_min_max, z_min_max, display=display)
    pcd1_filtered = filter_pcd(pcd1, x_min_max, y_min_max, z_min_max, display=display)

    # direct transformation calculated by RecFusion calibration pattern
    T = np.array([[0.618462, -0.428643,  0.658612,  -856.314/1000],
                  [0.428104,  0.886618,  0.175031,  -236.403/1000],
                  [-0.658963, 0.173705,  0.731843,   350.537/1000],
                  [-0,        -0,        -0,         1],
                  ])

    # direct transformation saved in gitlab
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    # T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957/1000],
    #               [0.45228962,    0.87499056,    0.17270096, -242.94946348/1000],
    #               [-0.64597401,   0.18787587,    0.73987852,  344.81816623/1000],
    #               [-0,        -0,        -0,         1],
    #               ])

    T = np.linalg.inv(T)  # use inverse transformation


    pcd0_filtered_T = copy.deepcopy(pcd0_filtered).transform(T)

    # display
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd 0', width=1600, height=1400)

    vis.add_geometry(pcd0_filtered_T)
    vis.add_geometry(pcd1_filtered)

    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()


    pass


def filter_pcd(pcd_in, x_min_max=[-1., 1.], y_min_max=[-1., 1.], z_min_max=[0., 1.5], display=False):

    points = np.asarray(pcd_in.points)
    ind = np.where((points[:, 0] > x_min_max[0]) & (points[:, 0] < x_min_max[1]) &
                   (points[:, 1] > y_min_max[0]) & (points[:, 1] < y_min_max[1]) &
                   (points[:, 2] > z_min_max[0]) & (points[:, 2] < z_min_max[1]))[0]

    pcd_filtered = copy.deepcopy(pcd_in)
    pcd_filtered = pcd_filtered.select_by_index(ind)

    dist_mean = 0.0012  # = calc_points_mean_dist(points, n_neighbors=5)  # e.g. 0.0012
    pcd_filtered = outlier_removal(pcd_filtered, dist_mean=dist_mean, radius_factor=20, nb_points=1000, iterations=1, display=display)

    if display:
        # change color of pcd_in
        color = [255. / 255, 140. / 255, 0. / 255]  # orange
        pcd_in.paint_uniform_color(color)

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='pcd filtered', width=1600, height=1400)
        vis.add_geometry(pcd_in)
        vis.add_geometry(pcd_filtered)
        opt = vis.get_render_option()
        opt.mesh_show_back_face = True

        vis.run()
        vis.destroy_window()

    return pcd_filtered


def outlier_removal(pcd_in, dist_mean=0.0012, radius_factor=15, nb_points=500, iterations=3, display=False):

    radius = radius_factor * dist_mean
    pcd_filtered = copy.deepcopy(pcd_in)
    ind_list = []
    for n in range(iterations):
        cl, ind = pcd_filtered.remove_radius_outlier(nb_points, radius)
        ind_list.append(ind)  # FIXME: need to treat different number of points in each iteration
        if display:
            display_inlier_outlier(pcd_filtered, ind)
        pcd_filtered = pcd_filtered.select_by_index(ind)

    #
    # ind_total = list(itertools.chain.from_iterable(ind_list))
    #
    # if display:
    #     display_inlier_outlier(pcd_in, ind_total)

    return pcd_filtered


def display_inlier_outlier(cloud, ind):
    """
    source: https://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html#Select-down-sample
    """
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    outlier_cloud.paint_uniform_color([1, 0, 0])
    # inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      # zoom=0.3412,
                                      # front=[0.4257, -0.2125, -0.8795],
                                      # lookat=[2.6172, 2.0475, 1.532],
                                      # up=[-0.0694, -0.9768, 0.2024],
                                      )
    pass

def calc_points_mean_dist(points, n_neighbors=5):

    if isinstance(points, o3d.cpu.pybind.geometry.PointCloud):
        points = np.asarray(points.points)

    n_neighbors = min(n_neighbors, len(points))  # n_neighbors cannot be larger than number of points

    # estimate average distance between points
    nbrs_dist_mean = NearestNeighbors(n_neighbors=n_neighbors, algorithm='kd_tree').fit(points)
    distances, indices = nbrs_dist_mean.kneighbors(points)
    dist_mean = distances[:, 1:].mean()  # exclude first entry

    return dist_mean




"""
{
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : false,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.066162109375, 0.941162109375, 3.0001518726160969 ],
			"boundingbox_min" : [ -1.866943359375, -1.153564453125, 0.49000003933921871 ],
			"field_of_view" : 60.0,
			"front" : [ -0.038879764680673251, -0.014974666807244292, -0.99913168464041202 ],
			"lookat" : [ 0.099609375, -0.106201171875, 1.7450759559776579 ],
			"up" : [ 0.13100167754733352, -0.99133410197675953, 0.0097600582844658227 ],
			"zoom" : 0.38199999999999967
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

"""

def pcd_registration():

    """
    based on:
    https://www.open3d.org/docs/release/tutorial/pipelines/multiway_registration.html
    """

    pcd0_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/00_points_0001.pcd'
    pcd1_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/01_points_0000.pcd'

    pcd0 = o3d.io.read_point_cloud(pcd0_file)
    pcd1 = o3d.io.read_point_cloud(pcd1_file)

    # direct transformation saved in gitlab
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957],
                  [0.45228962,    0.87499056,    0.17270096, -242.94946348],
                  [-0.64597401,   0.18787587,    0.73987852,  344.81816623],
                  [-0,        -0,        -0,         1],
                  ])

    pcd1_T = copy.deepcopy(pcd1).transform(T)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='pcd 0', width=1600, height=1400)

    vis.add_geometry(pcd0)
    # vis.add_geometry(pcd1)
    vis.add_geometry(pcd1_T)

    opt = vis.get_render_option()
    # opt.show_coordinate_frame = True
    vis.run()
    vis.destroy_window()

    # p0 = np.asarray(pcd0.points)
    # xy = p0[:, 0:2]
    # d = p0[:, 2].reshape(768, 1024)
    # cv2.imshow('depth', d)
    # cv2.waitKey(0)

    transformation_icp, information_icp = pairwise_registration(source=pcd1_T, target=pcd0)

    pcd1_T_2 = copy.deepcopy(pcd1_T).transform(transformation_icp)

    o3d.visualization.draw_geometries([pcd0, pcd1_T_2])

    pass


def pairwise_registration(source, target):
    print("Apply point-to-plane ICP")
    voxel_size = 1
    max_correspondence_distance_coarse = voxel_size * 750
    max_correspondence_distance_fine = voxel_size * 50

    source.estimate_normals()
    target.estimate_normals()

    icp_coarse = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_coarse, np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    icp_fine = o3d.pipelines.registration.registration_icp(
        source, target, max_correspondence_distance_fine,
        icp_coarse.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    return transformation_icp, information_icp

def run_reconstruction_system():

    config_json_file = None
    # config_json_file = 'config/realsense_l515.json'
    # default_dataset = 'lounge'
    # default_dataset = 'bedroom'
    default_dataset = 'IQ'
    debug_mode = True
    device = 'cpu:0'
    make = True  # make fragments
    register = True  # register fragments
    refine = True  # refine registration
    integrate = True  # integrate scene
    slac = False  # [Optional] Use --slac and --slac_integrate flags to perform SLAC optimisation.
    slac_integrate = False
    python_multi_threading = False

    if config_json_file is not None:
        with open(config_json_file) as json_file:
            config = json.load(json_file)
            initialize_config(config)
            check_folder_structure(config['path_dataset'])
    else:
        # load default dataset.
        config = dataset_loader(default_dataset)

    assert config is not None

    config['debug_mode'] = debug_mode
    config['device'] = device
    config['python_multi_threading'] = python_multi_threading

    print("====================================")
    print("Configuration")
    print("====================================")
    for key, val in config.items():
        print("%40s : %s" % (key, str(val)))

    times = [0, 0, 0, 0, 0, 0]
    if make:  # make fragments
        start_time = time.time()
        make_fragments.run(config)
        times[0] = time.time() - start_time
    if register:  # register fragments
        start_time = time.time()
        register_fragments.run(config)
        times[1] = time.time() - start_time
    if refine:  # refine registration
        start_time = time.time()
        refine_registration.run(config)
        times[2] = time.time() - start_time
    if integrate:  # integrate scene
        start_time = time.time()
        integrate_scene.run(config)
        times[3] = time.time() - start_time
    if slac:
        start_time = time.time()
        slac.run(config)
        times[4] = time.time() - start_time
    if slac_integrate:
        start_time = time.time()
        slac_integrate.run(config)
        times[5] = time.time() - start_time

    print("====================================")
    print("Elapsed time (in h:m:s)")
    print("====================================")
    print("- Making fragments    %s" % datetime.timedelta(seconds=times[0]))
    print("- Register fragments  %s" % datetime.timedelta(seconds=times[1]))
    print("- Refine registration %s" % datetime.timedelta(seconds=times[2]))
    print("- Integrate frames    %s" % datetime.timedelta(seconds=times[3]))
    print("- SLAC                %s" % datetime.timedelta(seconds=times[4]))
    print("- SLAC Integrate      %s" % datetime.timedelta(seconds=times[5]))
    print("- Total               %s" % datetime.timedelta(seconds=sum(times)))
    sys.stdout.flush()

    pass


def icp_playground():

    """
    Based on:
    https://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html#Point-to-plane-ICP
    """

    pcd0_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/00_points_0001.pcd'
    pcd1_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/01_points_0000.pcd'

    pcd0 = o3d.io.read_point_cloud(pcd0_file)
    pcd1 = o3d.io.read_point_cloud(pcd1_file)

    # direct transformation saved in gitlab
    # https://code.medtronic.com/magic_sw_and_algorithm_team/services/camera-service/-/blob/master/config/f1150179.cal?ref_type=heads
    T = np.array([[0.61494038,   -0.44619971,    0.65019547, -847.71289957],
                  [0.45228962,    0.87499056,    0.17270096, -242.94946348],
                  [-0.64597401,   0.18787587,    0.73987852,  344.81816623],
                  [-0,        -0,        -0,         1],
                  ])

    # pcd1_T = copy.deepcopy(pcd1).transform(T)

    voxel_size = 50

    source = copy.deepcopy(pcd1).voxel_down_sample(voxel_size=voxel_size)
    target = copy.deepcopy(pcd0).voxel_down_sample(voxel_size=voxel_size)
    trans_init = T

    draw_registration_result(source, target, trans_init)

    threshold = 500

    print("Initial alignment")
    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold, trans_init)
    print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)
    print("Transformation is:")
    print(reg_p2p.transformation)
    draw_registration_result(source, target, reg_p2p.transformation)

    print("Apply point-to-plane ICP")
    source.estimate_normals()
    target.estimate_normals()
    reg_p2l = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(reg_p2l)
    print("Transformation is:")
    print(reg_p2l.transformation)
    draw_registration_result(source, target, reg_p2l.transformation)

    pass

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def align_rgb_depth_example():

    rgb_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/00_color_0001.png'
    depth_file = '/Users/shilem2/data/rgbd/work_volume_data/20240409_154854_with_IQ_rec/00_depth_0001.png'

    output_dir = Path(rgb_file).parent / 'aligned_by_me'

    display = True
    # display = False
    force_rgb_z_1 = False

    rgb = cv2.imread(rgb_file, cv2.IMREAD_UNCHANGED)
    depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)

    # camera 00 - left camera (when standing in front of IQ)

    intrinsics_rgb = {'pp': [656.853, 346.649],
                      'f': [902.415, 902.806],
                      'w': 1280,
                      'h': 720,
                      'distortion_type': 'Brown Conrady',
                      'distortion': [0.152363, -0.457227, -7.3744e-06, 0.000388553, 0.419779],
                      }

    intrinsics_depth = {'pp': [500.984, 375.574],
                        'f': [737.297, 738.391],
                        'w': 1024,
                        'h': 768,
                        }
    depth_scale = 0.0002500000118743628

    R_depth2rgb = np.array([[0.999891, -0.0139051, -0.00503139],
                            [0.0137717, 0.999576, -0.02565],
                            [0.00538593, 0.0255779, 0.999658]])
    t_depth2rgb = np.array([[-0.000126142, 0.0140693, -0.002791]])
    T_depth2rgb = np.concatenate((R_depth2rgb, t_depth2rgb.T), axis=1)

    R_rgb2depth = np.array([[0.999891, 0.0137717, 0.00538593],
                            [-0.0139051, 0.999576, 0.0255779],
                            [-0.00503139, -0.02565, 0.999658]])
    t_rgb2depth = np.array([[0.000307721, -0.0141332, 0.00243087]])
    T_rgb2depth = np.concatenate((R_rgb2depth, t_rgb2depth.T), axis=1)

    # calculate rotation angles
    r = R.from_matrix(R_depth2rgb)
    euler_angles = r.as_euler('zyx', degrees=True)
    euler_delta = np.array([0, 0, 0])
    euler_angles_modified = euler_angles + euler_delta
    RR = R.from_euler('zyx', euler_angles_modified, degrees=True)
    R_depth2rgb = RR.as_matrix()
    T_depth2rgb = np.concatenate((R_depth2rgb, t_depth2rgb.T), axis=1)

    T = T_depth2rgb
    # T = T_rgb2depth

    # ppy_rgb_shift_list = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 50, 70]

    ppy_rgb_shift_list = [0]
    for ppy_rgb_shift in ppy_rgb_shift_list:
        rgb_aligned, depth_aligned = align_rgb_to_depth(rgb, intrinsics_rgb, depth, intrinsics_depth, depth_scale, T,
                                                        display=display,
                                                        ppy_rgb_shift=ppy_rgb_shift,
                                                        force_rgb_z_1=force_rgb_z_1,
                                                        )

        # save output
        output_dir.mkdir(exist_ok=True, parents=True)
        # cv2.imwrite((output_dir / Path(rgb_file).name).as_posix(), rgb_aligned)
        cv2.imwrite((output_dir / (Path(depth_file).stem + '_aligned' + Path(depth_file).suffix)).as_posix(), depth_aligned)
        cv2.imwrite((output_dir / (Path(depth_file).stem + '_orig' + Path(depth_file).suffix)).as_posix(), depth)

        sign_char = 'p' if ppy_rgb_shift > 0 else 'm'  # plus or minos
        rbg_aligned_file = (output_dir / (Path(rgb_file).stem + '_' + sign_char + str(abs(ppy_rgb_shift))
                                          + '_euler_delta_' + f'{euler_delta}'
                                          + Path(rgb_file).suffix)).as_posix()
        cv2.imwrite(rbg_aligned_file, rgb_aligned)

    pass

def align_rgb_to_depth(rgb, intrinsics_rgb, depth, intrinsics_depth, depth_scale, T_depth2rgb, display=False,
                       ppy_rgb_shift=0, force_rgb_z_1=False):
    """
    Adapted from:
    1) https://www.codefull.org/2016/03/align-depth-and-color-frames-depth-and-rgb-registration/
    2) Section 3 of : https://ap.isr.uc.pt/archive/clawar2011_v1.pdf
    """

    # unpack intrinsics
    fx_d = intrinsics_depth['f'][0]
    fy_d = intrinsics_depth['f'][1]
    ppx_d = intrinsics_depth['pp'][0]
    ppy_d = intrinsics_depth['pp'][1]
    h_depth = intrinsics_depth['h']
    w_depth = intrinsics_depth['w']

    fx_rgb = intrinsics_rgb['f'][0]
    fy_rgb = intrinsics_rgb['f'][1]
    ppx_rgb = intrinsics_rgb['pp'][0]
    ppy_rgb = intrinsics_rgb['pp'][1]
    h_rgb = intrinsics_rgb['h']
    w_rgb = intrinsics_rgb['w']
    # TODO: add distortion model? check if needed

    depth_transformed_coordinates = np.zeros((h_depth, w_depth, 3))
    depth_aligned = np.zeros((h_depth, w_depth), dtype=np.uint16)
    rgb_aligned = np.zeros((h_depth, w_depth, 3), dtype=np.uint8)

    for v in range(h_depth):
        for u in range(w_depth):

            # ------------
            # align depth
            # ------------
            # apply depth intrinsics
            z = depth[v, u] * depth_scale

            if force_rgb_z_1 and (z <= 0):
                z = 1.

            x = (u - ppx_d) * z / fx_d
            y = (v - ppy_d) * z / fy_d

            # apply extrinsics
            transformed = T_depth2rgb @ np.array([x, y, z, 1]).T

            x_t = transformed[0]
            y_t = transformed[1]
            z_t = transformed[2]

            depth_transformed_coordinates[v, u] = np.array([x_t, y_t, z_t])

            # apply depth intrinsics
            u_t = x_t * fx_d / z_t + ppx_d
            v_t = y_t * fy_d / z_t + ppy_d
            w_t = (z_t / depth_scale).round().astype(np.uint16)
            # FIXME: need to do interpolation, currently starting with simple rounding
            u_t = int(round(u_t))
            v_t = int(round(v_t))

            # depth_aligned[v_t, u_t] = w_t
            # depth_aligned[v, u] = w_t
            if (u_t > 0) and (u_t < w_depth) and (v_t > 0) and (v_t < h_depth):
                depth_aligned[v_t, u_t] = w_t

            pass

    # ------------
    # align rgb
    # ------------
    for v in range(h_depth):
        for u in range(w_depth):

            x_t, y_t, z_t = depth_transformed_coordinates[v, u]

            xx = x_t * fx_rgb / z_t + ppx_rgb
            yy = y_t * fy_rgb / z_t + ppy_rgb + ppy_rgb_shift
            # xx and yy are indices into the RGB frame, but they may contain invalid values of parts of the scene
            # that are not visible to the RGB camera
            if not np.isfinite(xx) or not np.isfinite(yy):
                continue
            # FIXME: need to do interpolation, currently starting with simple rounding
            xx = int(round(xx))
            yy = int(round(yy))
            if (xx < 0) or (xx >= w_rgb) or (yy < 0) or (yy >= h_rgb):
                continue

            # FIXME: need to do interpolation, currently starting with simple rounding
            rgb_aligned[v, u, 0] = rgb[yy, xx, 0]
            rgb_aligned[v, u, 1] = rgb[yy, xx, 1]
            rgb_aligned[v, u, 2] = rgb[yy, xx, 2]

            pass

    if display:
        cv2.imshow('rgb', rgb)
        cv2.imshow('depth', depth)
        cv2.imshow('rgb_aligned', rgb_aligned)
        cv2.imshow('depth_aligned', depth_aligned)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return rgb_aligned, depth_aligned


if __name__ == '__main__':

    # pcd_integration()
    pcd_integration_IQ_meshes()
    # pcd_registration()
    # run_reconstruction_system()
    # icp_playground()
    # align_rgb_depth_example()


    pass