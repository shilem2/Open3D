import numpy as np
import open3d as o3d
import copy
from pathlib import Path

"""
Adapted from
https://www.open3d.org/docs/release/tutorial/geometry/mesh.html#Mesh-filtering
"""

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


if __name__ == "__main__":

    mesh_processing_playground()

    pass