import numpy as np
import pyrealsense2.pyrealsense2 as rs

def initialize_camera():
    # Initialize the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.gyro)
    config.enable_stream(rs.stream.accel)
    pipeline.start(config)
    return pipeline

def get_camera_frames(pipeline):
    # Get camera frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    # Get the IMU frames
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    accel_frame = frames.first_or_default(rs.stream.accel)

    return depth_frame, color_frame, depth_image, color_image, gyro_frame, accel_frame


def stop_camera(pipeline):
    pipeline.stop()