from pathlib import Path

import copy

import cv2
import numpy as np
import open3d as o3d


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

    pcd0_T = copy.deepcopy(pcd0).transform(T)
    # pcd1_T = copy.deepcopy(pcd1).transform(T)

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

if __name__ == '__main__':

    # pcd_integration()
    # pcd_registration()
    # run_reconstruction_system()
    icp_playground()

    pass