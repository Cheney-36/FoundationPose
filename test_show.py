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

#import trimesh

from PIL import Image


if __name__=='__main__':
  
  # 初始化RealSense管道和配置
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

  # 启动管道
  profile = pipeline.start(config)

  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print("Depth Scale is: ", depth_scale)

  # 深度帧与其他帧对齐
  align_to = rs.stream.color
  align = rs.align(align_to)

      
  # 获取相机内参 ["color", "depth"]?
  camera_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

  # 获取相机内参并转换为NumPy数组
  K = np.array([[camera_intrinsics.fx, 0, camera_intrinsics.ppx],
                [0, camera_intrinsics.fy, camera_intrinsics.ppy],
                [0, 0, 1]])
  isfirst = True


  try:
      while True:
          frames = pipeline.wait_for_frames() 

          # Align the depth frame to color frame
          aligned_frames = align.process(frames)
          
          # Get aligned frames
          aligned_depth_frame = (aligned_frames.get_depth_frame())  
          color_frame = aligned_frames.get_color_frame()

          # Validate that both frames are valid
          if not aligned_depth_frame or not color_frame:
            continue

          
          depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)/1e3#??
          color_image = np.asanyarray(color_frame.get_data())
      
          ob_mask = Image.open(f'{code_dir}/demo_data/IndustReal/gear_assemb_20_40_60_base/mask_test.png')
          ob_mask_8bit = np.array(ob_mask)  # 这假设原始图像是归一化的（0到1的范围）

          cv2.imshow('Depth Image', depth_image)
          cv2.imshow('Color Image', color_image)
          cv2.imshow('Object Mask 8-bit', ob_mask_8bit)
          cv2.waitKey(1)#只有cv2.waitKey(1) RGB颜色才对   0接近全黑
  finally:
      pipeline.stop()

  # 清理
  cv2.destroyAllWindows()

