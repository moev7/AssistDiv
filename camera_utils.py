import numpy as np
import pyrealsense2.pyrealsense2 as rs
import time

def initialize_camera():
    # Initialize the RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) 
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    return pipeline


def get_camera_frames(pipeline, skip_frames=50):
    # Skip initial frames for camera to calibrate itself
    for _ in range(skip_frames):
        pipeline.wait_for_frames()

    # Get camera frames
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return depth_image, color_image


def stop_camera(pipeline):
    pipeline.stop()