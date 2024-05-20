import numpy as np
import open3d as o3d
import copy
from pathlib import Path
import matplotlib.pyplot as plt


def mesh_processing_playground():
    """
    Adapted from:
    https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Connected-components
    """

    mesh_file = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/scene/integrated_filtered.ply'

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # filter by connected components
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # print("Show mesh with small clusters removed")
    # mesh_0 = copy.deepcopy(mesh)
    # triangles_to_remove = cluster_n_triangles[triangle_clusters] < 100
    # mesh_0.remove_triangles_by_mask(triangles_to_remove)
    # o3d.visualization.draw_geometries([mesh_0], mesh_show_back_face=True)

    print("Show largest cluster")
    mesh_1 = copy.deepcopy(mesh)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_1.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_1], mesh_show_back_face=True)

    # save filtered mesh
    mesh_filtered_file = (Path(mesh_file).parent / (Path(mesh_file).stem + '_largest_cc.ply')).as_posix()
    o3d.io.write_triangle_mesh(mesh_filtered_file, mesh_1, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)


    pass


def mesh_smoothing_playground():
    """
    Adapted from:
    https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Mesh-filtering
    """

    mesh_file = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/scene/integrated_filtered_largest_cc.ply'

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # average filter
    number_of_iterations = 2
    mesh_average = copy.deepcopy(mesh).filter_smooth_simple(number_of_iterations=number_of_iterations)
    mesh_average.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh_average], mesh_show_back_face=True)
    mesh_filtered_file = (Path(mesh_file).parent / (Path(mesh_file).stem + f'_largest_cc_average_{number_of_iterations}_iters.ply')).as_posix()
    o3d.io.write_triangle_mesh(mesh_filtered_file, mesh_average, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

    # # taubin filter
    # mesh_taubin = copy.deepcopy(mesh).filter_smooth_taubin(number_of_iterations=1, lambda_filter=0.05)
    # mesh_taubin.compute_vertex_normals()
    # # mesh_out.paint_uniform_color([1, 0.706, 0])
    # o3d.visualization.draw_geometries([mesh_taubin], mesh_show_back_face=True)
    # mesh_filtered_file = (Path(mesh_file).parent / (Path(mesh_file).stem + '_largest_cc_taubin.ply')).as_posix()
    # o3d.io.write_triangle_mesh(mesh_filtered_file, mesh_taubin, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

    """
    Mesh simplification
    https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Mesh-simplification
    """
    print(f'Input mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles')

    # voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 128
    voxel_size = 0.001
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = copy.deepcopy(mesh).simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
    print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
    o3d.visualization.draw_geometries([mesh_smp], mesh_show_back_face=True)
    mesh_filtered_file = (Path(mesh_file).parent / (Path(mesh_file).stem + f'_largest_cc_simplified_voxel_size_{voxel_size}.ply')).as_posix()
    o3d.io.write_triangle_mesh(mesh_filtered_file, mesh_smp, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

    pass


def fill_holes_playground():

    mesh_file = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/scene/integrated_filtered_largest_cc.ply'
    mesh_file = '/Users/shilem2/data/rgbd/realsense_records/aligned_to_color/20240506_IQ/20240506_175654_IQ_left/scene/integrated_filtered_largest_cc_largest_cc_simplified_voxel_size_0.001.ply'

    mesh = o3d.io.read_triangle_mesh(mesh_file)
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # # fill_holes - not influence
    # # hole_size = 1000000.0  # default value is 1000000.0
    # hole_size = 1.0e20
    # mesh_filled = o3d.t.geometry.TriangleMesh.from_legacy(mesh).fill_holes(hole_size).to_legacy()
    # o3d.visualization.draw_geometries([mesh_filled], mesh_show_back_face=True)

    # get point cloud from mesh
    pcd = o3d.geometry.PointCloud(mesh.vertices)
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals
    # o3d.visualization.draw_geometries([pcd])

    # alpha shape
    # https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Alpha-shapes
    # alpha = 0.01
    # print(f"alpha={alpha:.3f}")
    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    # mesh.compute_vertex_normals()
    # o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)

    # Poisson surface reconstruction
    # https://www.open3d.org/docs/latest/tutorial/Advanced/surface_reconstruction.html#Poisson-surface-reconstruction
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_poison, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    o3d.visualization.draw_geometries([mesh_poison], mesh_show_back_face=True)

    # plot densities
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')((densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh_poison.vertices
    density_mesh.triangles = mesh_poison.triangles
    density_mesh.triangle_normals = mesh_poison.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh], mesh_show_back_face=True)

    # remove low density vertices
    quantile = 0.05
    mesh_poison_filtered = copy.deepcopy(mesh_poison)
    vertices_to_remove = densities < np.quantile(densities, quantile)
    mesh_poison_filtered.remove_vertices_by_mask(vertices_to_remove)
    o3d.visualization.draw_geometries([mesh_poison_filtered], mesh_show_back_face=True)

    # get largets connected component
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_poison_filtered.cluster_connected_triangles())
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    mesh_poison_filtered_cc = copy.deepcopy(mesh_poison_filtered)
    largest_cluster_idx = cluster_n_triangles.argmax()
    triangles_to_remove = triangle_clusters != largest_cluster_idx
    mesh_poison_filtered_cc.remove_triangles_by_mask(triangles_to_remove)
    o3d.visualization.draw_geometries([mesh_poison_filtered_cc], mesh_show_back_face=True)
    mesh_filtered_file = (Path(mesh_file).parent / (Path(mesh_file).stem + f'_poisson_rec_density_quantile_{quantile}_largest_cc.ply')).as_posix()
    o3d.io.write_triangle_mesh(mesh_filtered_file, mesh_poison_filtered_cc, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)

    # simplify mesh by vertex clustering
    voxel_size = 0.001
    print(f'voxel_size = {voxel_size:e}')
    mesh_smp = copy.deepcopy(mesh_poison_filtered_cc).simplify_vertex_clustering(voxel_size=voxel_size, contraction=o3d.geometry.SimplificationContraction.Average)
    print(f'Simplified mesh has {len(mesh_smp.vertices)} vertices and {len(mesh_smp.triangles)} triangles')
    o3d.visualization.draw_geometries([mesh_smp], mesh_show_back_face=True)
    mesh_filtered_file = (Path(mesh_filtered_file).parent / (Path(mesh_filtered_file).stem + f'_simplified_voxel_size_{voxel_size}.ply')).as_posix()
    o3d.io.write_triangle_mesh(mesh_filtered_file, mesh_smp, write_ascii=False, compressed=False, write_vertex_normals=True, write_vertex_colors=True, write_triangle_uvs=True, print_progress=False)


    pass


if __name__ == "__main__":

    # mesh_processing_playground()
    # mesh_smoothing_playground()
    fill_holes_playground()

    pass