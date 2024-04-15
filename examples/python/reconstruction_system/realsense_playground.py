import pyrealsense2 as rs


def get_camera_parameters():
    """
    Based on:
    https://dev.intelrealsense.com/docs/projection-texture-mapping-and-occlusion-with-intel-realsense-depth-cameras
    and more
    """

    # High resolution parameters
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipe = rs.pipeline()
    selection = pipe.start(config)

    # intrinsics
    depth_stream = selection.get_stream(rs.stream.depth).as_video_stream_profile()
    resolution = (depth_stream.width(), depth_stream.height())
    i_depth = depth_stream.get_intrinsics()
    print(i_depth)

    color_stream = selection.get_stream(rs.stream.color).as_video_stream_profile()
    resolution = (color_stream.width(), color_stream.height())
    i_color = color_stream.get_intrinsics()
    print(i_color)

    # extrinsics
    depth_to_color_extrinsics = depth_stream.get_extrinsics_to(selection.get_stream(rs.stream.color))
    print('depth_to_color_extrinsics')
    print(depth_to_color_extrinsics)

    color_to_depth_extrinsics = color_stream.get_extrinsics_to(selection.get_stream(rs.stream.depth))
    print('color_to_depth_extrinsics')
    print(color_to_depth_extrinsics)

    depth_sensor = selection.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print('depth_scale')
    print(depth_scale)

    # rs.align

    pass


if __name__ == '__main__':

    get_camera_parameters()

    pass