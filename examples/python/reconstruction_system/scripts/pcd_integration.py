from pathlib import Path

import copy

import cv2
import numpy as np
import open3d as o3d
import cv2

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
    display = False

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

    T = T_depth2rgb

    ppy_rgb_shift_list = [-70, -60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 50, 70]
    for ppy_rgb_shift in ppy_rgb_shift_list:
        rgb_aligned, depth_aligned = align_rgb_to_depth(rgb, intrinsics_rgb, depth, intrinsics_depth, depth_scale, T, display=display, ppy_rgb_shift=ppy_rgb_shift)

        # save output
        output_dir.mkdir(exist_ok=True, parents=True)
        # cv2.imwrite((output_dir / Path(rgb_file).name).as_posix(), rgb_aligned)
        # # cv2.imwrite((output_dir / Path(depth_file).name).as_posix(), depth_aligned)
        cv2.imwrite((output_dir / Path(depth_file).name).as_posix(), depth)

        sign_char = 'p' if ppy_rgb_shift > 0 else 'm'  # plus or minos
        rbg_aligned_file = (output_dir / (Path(rgb_file).stem + '_' + sign_char + str(abs(ppy_rgb_shift)) + Path(rgb_file).suffix)).as_posix()
        cv2.imwrite(rbg_aligned_file, rgb_aligned)

    pass

def align_rgb_to_depth(rgb, intrinsics_rgb, depth, intrinsics_depth, depth_scale, T_depth2rgb, display=False,
                       ppy_rgb_shift=0):
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

    depth_aligned = np.zeros((h_depth, w_depth, 3))
    rgb_aligned = np.zeros((h_depth, w_depth, 3), dtype=np.uint8)

    for v in range(h_depth):
        for u in range(w_depth):

            # ------------
            # align depth
            # ------------
            # apply depth intrinsics
            # z = 1  # depth[v, u] * depth_scale
            z = depth[v, u] * depth_scale
            x = (u - ppx_d) * z / fx_d
            y = (v - ppy_d) * z / fy_d

            # apply extrinsics
            transformed = T_depth2rgb @ np.array([x, y, z, 1]).T
            depth_aligned[v, u, 0] = transformed[0]
            depth_aligned[v, u, 1] = transformed[1]
            depth_aligned[v, u, 2] = transformed[2]

            # ------------
            # align rgb
            # ------------
            xx = x * fx_rgb / z + ppx_rgb
            yy = y * fy_rgb / z + ppy_rgb + ppy_rgb_shift
            # xx and yy are indices into the RGB frame, but they may contain invalid values of parts of the scene
            # that are not visible to the RGB camera
            if not np.isfinite(xx) or not np.isfinite(yy):
                continue
            # FIXME: need to do interpolation, currently starting with simple rounding
            xx = int(round(xx))
            yy = int(round(yy))
            if (xx < 0) or (xx >= w_rgb) or (yy < 0) or (yy >= h_rgb) :
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
    # pcd_registration()
    # run_reconstruction_system()
    # icp_playground()
    align_rgb_depth_example()

    pass