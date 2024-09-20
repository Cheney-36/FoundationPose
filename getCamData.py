import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os

pipeline = rs.pipeline()

#Create a config并配置要流​​式传输的管道
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)




# 获取相机 RGB内参
camera_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
print("RGB:   ",camera_intrinsics) 

# 获取相机 D内参
camera_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
print("D:   ",camera_intrinsics) 


align_to = rs.stream.color
align = rs.align(align_to)

# 按照日期创建文件夹
save_path = os.path.join(os.getcwd(), "out", time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
os.mkdir(save_path)
os.mkdir(os.path.join(save_path, "color"))
os.mkdir(os.path.join(save_path, "depth"))

# 保存的图片和实时的图片界面
cv2.namedWindow("live", cv2.WINDOW_AUTOSIZE)
cv2.namedWindow("save", cv2.WINDOW_AUTOSIZE)
saved_color_image = None # 保存的临时图片
saved_depth_mapped_image = None
saved_count = 0

# 主循环
try:
    while True:
        frames = pipeline.wait_for_frames()

        aligned_frames = align.process(frames)

        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not color_frame:
            continue
        
        depth_data = np.asanyarray(aligned_depth_frame.get_data(), dtype="float16")
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 自动调整深度数据到0-65535范围
        min_val, max_val = depth_data.min(), depth_data.max()
        depth_image_scaled = np.clip((depth_data - min_val) / (max_val - min_val) * 65535, 0, 65535).astype(np.uint16)

        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        cv2.imshow("live", np.hstack((color_image, depth_mapped_image)))

        # OpenCV不需要对数据进行归一化就可以直接保存为EXR格式
        depth_image_exr = (depth_data).astype(np.float32)  # 将深度数据转换为实际的深度值（米）

        key = cv2.waitKey(30)

        # s 保存图片
        if key & 0xFF == ord('s'):
            saved_color_image = color_image
            saved_depth_mapped_image = depth_mapped_image

            # 彩色图片保存为png格式
            cv2.imwrite(os.path.join((save_path), "color", "{}.png".format(saved_count)), saved_color_image)
            # # 深度信息由采集到的float16直接保存为npy格式
            # np.save(os.path.join((save_path), "depth", "{}".format(saved_count)), depth_data)
             # 深度图像以16位PNG格式保存
            cv2.imwrite(os.path.join((save_path), "depth", "{}.png".format(saved_count)), depth_image_scaled)
            # 深度图像以EXR格式保存
            cv2.imwrite(os.path.join((save_path), "depth", "{}.exr".format(saved_count)), depth_image_exr)

            saved_count+=1
            cv2.imshow("save", np.hstack((saved_color_image, saved_depth_mapped_image)))

        # q 退出
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break    
finally:
    pipeline.stop()