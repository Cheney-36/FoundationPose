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

from mask import create_mask
#import trimesh

from PIL import Image


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))

  parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/IndustReal/gear_assemb_20_40_60_base/meshes/cube40mm.obj')
  #parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/IndustReal/gear_assemb_20_40_60_base/meshes/link1.obj')

  parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  parser.add_argument('--est_refine_iter', type=int, default=5)
  parser.add_argument('--track_refine_iter', type=int, default=2)
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  # 加载3D模型
  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  # 初始化位姿估计器
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()

  # 根据3D模型初始化位姿估计器
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")


  # 定义帧率
  fps = 30
  # 定义图像的宽度和高度
  width = 720
  height = 1280
  # 初始化RealSense管道和配置
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.color, height, width, rs.format.bgr8, fps)
  config.enable_stream(rs.stream.depth, height, width, rs.format.z16, fps)

  logging.info("如果需要重新制作mask,请输入1，否则其他....")
  if input() == "1":
    create_mask(height,width,fps)


  # 启动管道
  profile = pipeline.start(config)

  depth_sensor = profile.get_device().first_depth_sensor()
  depth_scale = depth_sensor.get_depth_scale()
  print("Depth Scale is: ", depth_scale)

  # 深度帧与其他帧对齐
  align_to = rs.stream.color
  align = rs.align(align_to)

  # 获取相机内参 对齐之后是以color为基准的
  camera_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
  # 获取相机内参并转换为NumPy数组
  K = np.array([[camera_intrinsics.fx, 0, camera_intrinsics.ppx],
                [0, camera_intrinsics.fy, camera_intrinsics.ppy],
                [0, 0, 1]])
  isfirst = True
  print("K:", K)
  # 位姿估计循环
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
          
          # 原始深度图像转换为米
          depth_image = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32) * depth_scale
          color_image = np.asanyarray(color_frame.get_data())

          # H, W = cv2.resize(color_image, (640,480)).shape[:2]
          # color_image = cv2.resize(color_image, (W,H), interpolation=cv2.INTER_NEAREST)
          # depth_image = cv2.resize(depth_image, (W,H), interpolation=cv2.INTER_NEAREST)
      
          if isfirst:
            mask = cv2.imread("mask.png")
            if len(mask.shape)==3:
                for c in range(3):
                    if mask[...,c].sum()>0:
                        mask = mask[...,c]
                        break

            ob_mask_8bit = np.array(mask).astype(bool) #.astype(np.uint8)  # 这假设原始图像是归一化的（0到1的范围）

            pose = est.register(K=K, rgb=color_image, depth=depth_image, ob_mask=ob_mask_8bit, iteration=args.est_refine_iter)
            isfirst=False
          else:
            pose = est.track_one(K=K, rgb=color_image, depth=depth_image,iteration=args.track_refine_iter)
          
          # 调试输出
          if args.debug >= 1:
            center_pose = pose@np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(K, img=color_image, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color_image, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imshow('1', vis[...,::-1])
            cv2.waitKey(1)

  finally:
      pipeline.stop()

  # 清理
  cv2.destroyAllWindows()




