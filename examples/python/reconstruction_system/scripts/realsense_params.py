import pyrealsense2 as rs
import numpy as np
import cv2
from pathlib import Path
import datetime
import json


def convert_array_to_json(x):
    if hasattr(x, "tolist"):  # numpy arrays have this
        return {"$array": x.tolist()}  # Make a tagged object
    if isinstance(x, datetime.date):
        return {"$date": x.isoformat()}
    raise TypeError(x)

def convert_json_to_array(x):
    if len(x) == 1:  # Might be a tagged object...
        key, value = next(iter(x.items()))  # Grab the tag and value
        if key == "$array":  # If the tag is correct,
            return np.array(value)  # cast back to array
        if key == "$date":
            return datetime.datetime.strptime(value, '%Y-%m-%d').date()
    return x

def get_camera_params(rs, selection, out_json_path=None):

    # device
    device = selection.get_device()
    device_name = device.get_info(rs.camera_info.name)
    serial_number = device.get_info(rs.camera_info.serial_number)
    firmware_version = device.get_info(rs.camera_info.firmware_version)

    # intrinsics
    depth_stream = selection.get_stream(rs.stream.depth).as_video_stream_profile()
    intrinsics_depth = depth_stream.get_intrinsics()

    color_stream = selection.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics_color = color_stream.get_intrinsics()

    # extrinsics
    depth_to_color_extrinsics = depth_stream.get_extrinsics_to(selection.get_stream(rs.stream.color))
    color_to_depth_extrinsics = color_stream.get_extrinsics_to(selection.get_stream(rs.stream.depth))

    depth_sensor = selection.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    cam_params = {
        'device': {'name': device_name,
                   'serial_number': serial_number,
                   'firmware_version': firmware_version,
                   },
        'depth': {'intrinsics': str(intrinsics_depth),
                  'width': intrinsics_depth.width,
                  'height': intrinsics_depth.height,
                  'fx': intrinsics_depth.fx,
                  'fy': intrinsics_depth.fy,
                  'ppx': intrinsics_depth.ppx,
                  'ppy': intrinsics_depth.ppy,
                  'model': str(intrinsics_depth.model),
                  'coeffs': intrinsics_depth.coeffs,
                  'depth_scale': depth_scale,
                  'format': str(depth_stream.format),
                  },
        'color': {'intrinsics': str(intrinsics_color),
                  'width': intrinsics_color.width,
                  'height': intrinsics_color.height,
                  'fx': intrinsics_color.fx,
                  'fy': intrinsics_color.fy,
                  'ppx': intrinsics_color.ppx,
                  'ppy': intrinsics_color.ppy,
                  'model': str(intrinsics_color.model),
                  'coeffs': intrinsics_color.coeffs,
                  'format': str(color_stream.format),
                  },
        'transformations': {'depth_to_color_extrinsics': {'rotation': depth_to_color_extrinsics.rotation,
                                                          'translation': depth_to_color_extrinsics.translation,
                                                          },
                            'color_to_depth_extrinsics': {'rotation': color_to_depth_extrinsics.rotation,
                                                          'translation': color_to_depth_extrinsics.translation,
                                                          },
                            },
    }

    cam_params_json = json.dumps(cam_params, default=convert_array_to_json, indent=4, sort_keys=True)

    if out_json_path is not None:
        out_json_path = Path(out_json_path)
        out_json_path.parent.mkdir(exist_ok=True, parents=True)
        with open(out_json_path.resolve(), 'w+') as f:
            f.write(cam_params_json)

    return cam_params, cam_params_json


if __name__ == '__main__':

    config = rs.config()
    config.enable_stream(rs.stream.depth, 1024, 768, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipe = rs.pipeline()
    selection = pipe.start(config)

    cam_params_file = Path('__file__').parent / 'cam_params.json'

    cam_params, cam_params_json = get_camera_params(rs, selection, cam_params_file)

    print(cam_params_json)

    pass