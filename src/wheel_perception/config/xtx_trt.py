import os
import subprocess

def build_engine(onnx_file_path, engine_file_path, trtexec_path):
    if not os.path.exists(onnx_file_path):
        print(f"ONNX 文件不存在: {onnx_file_path}")
        return False
        
    if not os.path.exists(trtexec_path):
        print(f"trtexec 不存在: {trtexec_path}")
        return False

    print("\n[3/4] 编译为 TensorRT 引擎...")
    
    cmd = [
        trtexec_path,
        f"--onnx={onnx_file_path}",
        f"--saveEngine={engine_file_path}",
        "--fp16", # 启用 FP16 加速
        "--workspace=1024" # 分配 1GB 工作空间 (旧版本TRT适用)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"\nEngine 文件已成功保存至: {engine_file_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"构建 Engine 失败: {e}")
        return False

if __name__ == "__main__":
    # 配置输入和输出路径
    onnx_path = "/home/tensorlab/alphawheel/alpha_ws/src/wheel_perception/config/xtx-roadseg_rgb.onnx"
    engine_path = "/home/tensorlab/alphawheel/alpha_ws/src/wheel_perception/config/xtx-roadseg_rgb.engine"
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    
    build_engine(onnx_path, engine_path, trtexec)