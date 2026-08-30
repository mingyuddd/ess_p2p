#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import sys
import subprocess
from unittest.mock import MagicMock

# 建议在运行前确保已安装: pip install onnx onnxsim
try:
    import onnx
    from onnxsim import simplify
    HAS_ONNXSIM = True
except ImportError:
    HAS_ONNXSIM = False
    print("警告: 未检测到 onnxsim。强烈建议运行 'pip install onnx onnxsim' 以优化计算图提升帧率！")

# ================= 新增补丁开始 =================
def custom_unflatten(self, dim, sizes):
    shape = list(self.shape)
    if dim < 0:
        dim += len(shape)
    new_shape = shape[:dim] + list(sizes) + shape[dim+1:]
    return self.reshape(new_shape)

torch.Tensor.unflatten = custom_unflatten
# ================= 新增补丁结束 =================

# 解决 mmengine 导入时因 torch 缺乏分布式支持而引发的一系列报错
sys.modules['mmengine.model.wrappers.fully_sharded_distributed'] = MagicMock()

if not hasattr(torch.distributed, 'ReduceOp'):
    class ReduceOpMock:
        SUM = 0
    torch.distributed.ReduceOp = ReduceOpMock

if not hasattr(torch.distributed, 'group'):
    class GroupMock:
        WORLD = None
    torch.distributed.group = GroupMock

# 把已有的 mmsegmentation 源码目录强行加入 Python 环境变量
sys.path.append("/home/tensorlab/alphawheel/wheel_together/src/wheel/scripts/mmsegmentation-main")

from mmseg.apis import init_model
print("✓ 成功导入 mmsegmentation")

class MMSegInference(nn.Module):
    """MMSegmentation 模型前向包装器 - 只输出推理 logits"""
    def __init__(self, config_path, pth_path, device='cuda', target_size=(720, 1280)):
        super().__init__()
        # 强制在 GPU 初始化模型
        self.model = init_model(config_path, pth_path, device=device)
        self.model.eval()
        self.target_size = target_size
    
    def forward(self, x):
        """
        处理 mmseg 前向推断，并严格对齐输出分辨率
        """
        out = None
        if hasattr(self.model, 'forward'):
            try:
                out = self.model(inputs=x, mode='tensor')
            except Exception:
                pass
                
        if out is None and hasattr(self.model, 'encode_decode'):
            try:
                out = self.model.encode_decode(x, img_metas=None)
            except Exception:
                pass
            
        if out is None:
            out = self.model(x)

        # [性能修复]：使用静态的写死 size 而非动态获取 x.shape 的结果。
        # 避免给 ONNX 送入例如 Shape/Gather 等动态读取节点，大幅度提升 TRT 处理效率。
        # 必须确保 align_corners 为 False 或与训练一致。
        import torch.nn.functional as F
        out = F.interpolate(out, size=self.target_size, mode='bilinear', align_corners=False)

        return out

def mmseg_pth_to_engine(config_path, pth_path, engine_path, input_size=(720, 1280)):
    print("=" * 60)
    print("MMSegmentation (PyTorch) → TensorRT 转换 (GPU 加速版)")
    print("=" * 60)
    
    temp_onnx = "/tmp/mmseg_temp_model.onnx"
    temp_onnx_sim = "/tmp/mmseg_temp_model_sim.onnx"
    
    # 强制在 GPU 上进行推理和导出
    if not torch.cuda.is_available():
        print("✗ 严重警告: 当前未检测到可用的 GPU (CUDA)，将退回 CPU 导出！")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        print(f"✓ 已绑定转换设备: {torch.cuda.get_device_name(0)}")
    
    # 1. 加载模型
    print("\n[1/4] 解析 Config 并加载 PyTorch 模型权重...")
    if not os.path.exists(config_path):
        print(f"✗ 找不到 Config 文件: {config_path}")
        return False
        
    try:
        wrapper = MMSegInference(config_path, pth_path, device=device, target_size=input_size)
        print(f"✓ 模型加载成功，参数量: {sum(p.numel() for p in wrapper.model.parameters())/1e6:.2f}M")
    except Exception as e:
        print(f"✗ 模型加载失败:\n{e}")
        return False
        
    # 2. 导出ONNX
    print("\n[2/4] 导出 ONNX 中 (这可能需要一两分钟)...")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1], requires_grad=False).to(device)
    
    try:
        # 去掉 torch.cuda.amp.autocast()，保持纯 FP32 导出
        with torch.no_grad(): 
            torch.onnx.export(
                wrapper,
                dummy_input,
                temp_onnx,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output']
            )
        print(f"✓ 原图 ONNX 导出成功: {temp_onnx}")
    except Exception as e:
        print(f"✗ ONNX 导出失败:\n{e}")
        return False

    # 3. 使用 ONNX Simplifier 化简网络 (极大提升排坑能力和算子融合能力)
    onnx_for_trt = temp_onnx
    if HAS_ONNXSIM:
        print("\n[化简] 正在执行 ONNX 结构化简与冗余算子消除...")
        try:
            model_onnx = onnx.load(temp_onnx)
            model_simp, check = simplify(model_onnx)
            if check:
                onnx.save(model_simp, temp_onnx_sim)
                onnx_for_trt = temp_onnx_sim
                print(f"✓ 算子图化简成功，去除了碎片节点。保存为: {temp_onnx_sim}")
            else:
                print("⚠ 算子图化简校验失败，将使用未化简的 ONNX")
        except Exception as e:
            print(f"⚠ 执行简化发生错误，降级使用未简化版本: {e}")
        
    # 4. 转换为 TensorRT 引擎
    print("\n[3/4] 编译为 TensorRT 引擎...")
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    
    if not os.path.exists(trtexec):
        print(f"✗ 找不到 trtexec 可执行文件: {trtexec}")
        return False
        
    cmd = [
        trtexec,
        f"--onnx={onnx_for_trt}",
        f"--saveEngine={engine_path}",
        "--memPoolSize=workspace:2048", 
        "--fp16",              
        "--verbose"            
    ]
    
    print("正在执行编译命令: \n" + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 清理现场
    if os.path.exists(temp_onnx): os.remove(temp_onnx)
    if os.path.exists(temp_onnx_sim): os.remove(temp_onnx_sim)
        
    if result.returncode == 0:
        print("\n[4/4] 转换完成!")
        print("=" * 60)
        print(f"✅ 成功编译 Engine!")
        print(f"输出引擎: {engine_path}")
        print(f"引擎大小: {os.path.getsize(engine_path)/1024/1024:.2f} MB")
        print("=" * 60)
        return True
    else:
        print("\n❌ 转换失败! trtexec 报错信息如下:")
        print(result.stderr[-3000:]) 
        return False

if __name__ == "__main__":
    config_file = "/home/tensorlab/alphawheel/wheel_together/src/wheel/scripts/mmsegmentation-main/configs/segformer/segformer_mit-b0_8xb1-160k_cityscapes-1024x1024.py"
    pth_file = "/home/tensorlab/alphawheel/wheel_cuda/src/wheel_perception/config/b0_city1lakes2.pth"
    engine_file = "/home/tensorlab/alphawheel/wheel_cuda/src/wheel_perception/config/sealackcityb0_480.engine"
    
    if not os.path.exists(pth_file):
        print(f"提醒: 找不到 pth 文件 {pth_file}，请修改 __main__ 中的文件路径。")
        sys.exit(1)
        
    success = mmseg_pth_to_engine(
        config_path=config_file,
        pth_path=pth_file,
        engine_path=engine_file,
        input_size=(480,720)   
    )
