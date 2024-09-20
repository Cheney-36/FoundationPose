import pyrealsense2 as rs
import yaml

def save_intrinsics_and_distortion_as_yaml(width, height, fps):
    # 初始化 pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # 配置 RGB 和 深度流
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

    try:
        # 启动流
        profile = pipeline.start(config)

        # 获取内参和畸变参数数据
        intrinsics_files = {}
        streams = ["color", "depth"]

        for stream in streams:
            stream_profile = profile.get_stream(getattr(rs.stream, stream)).as_video_stream_profile()
            intrinsics = stream_profile.get_intrinsics()
            
            # 准备 YAML 格式的内参和畸变数据
            camera_info = {
                "camera_matrix": {
                    "rows": 3,
                    "cols": 3,
                    "dt": "d",
                    "data": [
                        intrinsics.fx, 0.0, intrinsics.ppx,
                        0.0, intrinsics.fy, intrinsics.ppy,
                        0.0, 0.0, 1.0
                    ]
                },
                "distortion_coefficients": {
                    "rows": 1,
                    "cols": 5,
                    "dt": "d",
                    "data": intrinsics.coeffs
                }
            }
            
            # 将内参和畸变参数数据保存到 YAML 文件
            file_name = f"{stream}_intrinsics_and_distortion.yaml"
            with open(file_name, 'w') as file:
                yaml.dump(camera_info, file, default_flow_style=None, sort_keys=False)

            intrinsics_files[stream] = file_name

        return intrinsics_files
    finally:
        # 停止 pipeline
        pipeline.stop()

# 使用示例
intrinsics_files = save_intrinsics_and_distortion_as_yaml(640, 480, 30)
print("Intrinsics and distortion files saved:", intrinsics_files)
