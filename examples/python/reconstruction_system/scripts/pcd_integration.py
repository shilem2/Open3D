from pathlib import Path

import copy
import numpy as np
import open3d as o3d


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
    # T = np.array([[0.618462,  0.428104, -0.658963,   861.793],
    #               [-0.428643, 0.886618,  0.173705,  -218.344],
    #               [0.658612,  0.175031,  0.731843,   348.819],
    #               [-0,        -0,        -0,         1],
    #               ])

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
    opt.mesh_show_back_face = True
    vis.run()
    vis.destroy_window()

    pass


if __name__ == '__main__':

    pcd_integration()

    pass