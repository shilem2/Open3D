## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


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
    print(selection.get_device())

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


def align_depth2color_example():
    """
    Adapted from:
    https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py
    """

    align_to_stream = 'color'  # color or depth
    # align_to_stream = 'depth'  # color or depth

    assert align_to_stream in ['color', 'depth']

    output_root = 'C:/projects/rgbd/data/realsense_records/'

    output_sfx = '_first_try'

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_root) / f'aligned_to_{align_to_stream}' / (date_str + output_sfx)

    depth_dir = output_dir / 'depth'
    color_dir = output_dir / 'color'

    depth_dir.mkdir(exist_ok=True, parents=True)
    color_dir.mkdir(exist_ok=True, parents=True)

    # Create a pipeline
    pipeline = rs.pipeline()

    # Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))

    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    # config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)


    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is: " , depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    clipping_distance_in_meters = 3 #1 meter
    clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    if align_to_stream == 'color':
        align_to = rs.stream.color
    elif align_to_stream == 'depth':
        align_to = rs.stream.depth

    align = rs.align(align_to)

    # Streaming loop
    counter = -1
    try:
        while True:
            counter += 1
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # cv2.imshow('color', color_image)
            # cv2.waitKey(0)

            filename = f'{counter:05}.png'
            cv2.imwrite((depth_dir / filename).as_posix(), depth_image)
            cv2.imwrite((color_dir / filename).as_posix(), color_image)

            # Remove background - Set pixels further than clipping_distance to grey
            # grey_color = 153
            # depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #depth image is 1 channel, color is 3 channels
            # bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color, color_image)

            # Render images:
            #   depth align to color on left
            #   depth on right
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            images = np.hstack((color_image, depth_colormap))

            cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
            cv2.imshow('Align Example', images)
            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()

    pass


if __name__ == '__main__':

    align_depth2color_example()
    get_camera_parameters()

    pass