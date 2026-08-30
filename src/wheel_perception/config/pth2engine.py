#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import os
import sys
import subprocess
import numpy as np

# 添加BiSeNet路径
bisenet_path = "/home/smy/wheel_ess/BiSeNet"
if bisenet_path not in sys.path:
    sys.path.insert(0, bisenet_path)

try:
    from lib.models.bisenetv2 import BiSeNetV2
    print("✓ 成功导入BiSeNetV2")
except Exception as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

class BiSeNetV2Inference(nn.Module):
    """推理模型 - 只输出主logits"""
    def __init__(self, n_classes=19):
        super().__init__()
        self.model = BiSeNetV2(n_classes=n_classes, aux_mode='eval')
    
    def forward(self, x):
        logits, *_ = self.model(x)
        return logits
    
    def load_weights(self, pth_path):
        """加载权重"""
        checkpoint = torch.load(pth_path, map_location='cpu')
        
        # 处理权重字典
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 移除 'module.' 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.model.eval()
        return self

def pth_to_engine(pth_path, engine_path, input_size=(512, 1024), num_classes=19):
    """
    直接将.pth转换为TensorRT引擎
    
    Args:
        pth_path: .pth文件路径
        engine_path: 输出的.engine文件路径
        input_size: 输入尺寸 (height, width)
        num_classes: 类别数
    """
    print("=" * 60)
    print("PyTorch → TensorRT 一键转换")
    print("=" * 60)
    
    # 1. 创建临时ONNX文件
    temp_onnx = "/tmp/temp_model.onnx"
    
    # 2. 创建并加载模型
    print("\n[1/4] 加载PyTorch模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiSeNetV2Inference(num_classes)
    model.load_weights(pth_path)
    model.to(device)
    model.eval()
    print(f"✓ 模型加载成功，参数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 3. 导出ONNX
    print("\n[2/4] 导出ONNX...")
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1]).to(device)
    
    torch.onnx.export(
        model,
        dummy_input,
        temp_onnx,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )
    print(f"✓ ONNX导出成功: {temp_onnx}")
    
    # 4. 转换为TensorRT引擎
    print("\n[3/4] 转换为TensorRT引擎...")
    # [本机适配] 无系统 trtexec，使用 TRT 10 构建器（build_all_trt10.sh 会先编译）
    trtexec = "/home/smy/wheel_ess/src/wheel_perception/ess/build_engine_trt10"

    if not os.path.exists(trtexec):
        print(f"✗ 找不到构建器: {trtexec}")
        print("请先运行: bash /home/smy/wheel_ess/src/wheel_perception/ess/build_all_trt10.sh")
        return False

    cmd = [
        trtexec,
        f"--onnx={temp_onnx}",
        f"--saveEngine={engine_path}",
        "--fp16",
        f"--input-shapes=input:1x3x{input_size[0]}x{input_size[1]}"
    ]
    
    print("执行: " + " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # 5. 清理临时文件
    if os.path.exists(temp_onnx):
        os.remove(temp_onnx)
    
    if result.returncode == 0:
        print("\n[4/4] 完成!")
        print("=" * 60)
        print(f"✅ 转换成功!")
        print(f"输入: {pth_path}")
        print(f"输出: {engine_path}")
        print(f"引擎大小: {os.path.getsize(engine_path)/1024/1024:.2f} MB")
        print("=" * 60)
        return True
    else:
        print("\n❌ 转换失败!")
        print(result.stderr)
        return False

if __name__ == "__main__":
    # 配置参数
    pth_file = "/home/smy/wheel_ess/src/wheel_perception/config/bisenet_combined4.pth"
    # [本机适配] 输出到 params.yaml 配置的实际路径
    engine_file = "/home/smy/wheel_ess/src/wheel_perception/config/combined4.engine"
    
    # 检查文件是否存在
    if not os.path.exists(pth_file):
        # 尝试备用路径
        alt_file = "/home/smy/wheel_ess/src/wheel_perception/config/bisenet_combined4.pth"
        if os.path.exists(alt_file):
            pth_file = alt_file
            print(f"使用备用路径: {pth_file}")
        else:
            print(f"错误: 找不到 {pth_file}")
            sys.exit(1)
    
    # 执行转换
    success = pth_to_engine(
        pth_path=pth_file,
        engine_path=engine_file,
        input_size=(512, 1024),  # 可根据需要调整
        num_classes=19  # Cityscapes是19类，如果是二分类请改为2
    )
    
    if success:
        print(f"\n现在可以在params.yaml中设置:")
        print(f"engine_path: \"{os.path.abspath(engine_file)}\"")
