# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from estimater import *
from datareader import *
import argparse

from pyrealsense2 import pyrealsense2 as rs# RealSense SDK
import os
import logging
import numpy as np
import cv2  # OpenCV库，用于处理图像
import sys


if __name__=='__main__':
  # 初始化RealSense管道和配置
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

  # 启动管道
  profile = pipeline.start(config)

  # 获取相机内参
  camera_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

  #test can getImage demo
  while True:
        frames = pipeline.wait_for_frames()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.array(depth_frame.get_data(), dtype=np.float32)

        #尺度转换和绝对值变换
        scaled_depth=cv2.convertScaleAbs(depth_image, alpha=0.08)
        #应用颜色映射
        depth_colormap = cv2.applyColorMap(scaled_depth, cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.imshow('RealSense', images)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
  pipeline.stop()
  cv2.destroyAllWindows()
  


