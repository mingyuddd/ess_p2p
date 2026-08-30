#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import argparse
import shutil

# ---------- 依赖检查 ----------
try:
    import onnx
    from onnxsim import simplify
except ImportError as e:
    print("❌ 缺少必要的 Python 包: onnx 和 onnx-simplifier")
    print("   请执行: pip install onnx onnx-simplifier")
    sys.exit(1)

def find_trtexec():
    """自动查找 trtexec 可执行文件路径"""
    # 1. 用户脚本中曾经指定的路径
    candidates = [
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/bin/trtexec",
        "/opt/nvidia/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec"
    ]
    for path in candidates:
        if os.path.exists(path) and os.access(path, os.X_OK):
            return path
    # 2. 从环境变量 PATH 中查找
    trtexec = shutil.which("trtexec")
    if trtexec:
        return trtexec
    return None

def simplify_onnx_model(onnx_path, output_path=None):
    """
    使用 onnx-simplifier 简化 ONNX 模型，展开融合算子（如 FusedConcatConv3x3）
    """
    if output_path is None:
        base, ext = os.path.splitext(onnx_path)
        output_path = f"{base}_simplified{ext}"

    print("🔧 正在简化 ONNX 模型（展开融合算子）...")
    try:
        model = onnx.load(onnx_path)
        model_simplified, check = simplify(model)
        if not check:
            print("⚠️  简化失败，将使用原始模型（可能导致转换错误）")
            return onnx_path
        onnx.save(model_simplified, output_path)
        print(f"✅ 简化成功，简化后的模型保存为: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ 简化过程出错: {e}")
        return None

def onnx_to_engine(onnx_path, engine_path, workspace_mb=4096, fp16=True, verbose=True, use_simplify=True):
    """
    使用 trtexec 将 ONNX 转换为 TensorRT 引擎
    """
    # 1. 检查输入 ONNX 文件
    if not os.path.exists(onnx_path):
        print(f"❌ ONNX 文件不存在: {onnx_path}")
        return False

    # 2. 可选的简化步骤
    if use_simplify:
        simplified_onnx = simplify_onnx_model(onnx_path)
        if simplified_onnx is None:
            print("⚠️  简化失败，继续使用原始 ONNX")
            simplified_onnx = onnx_path
        onnx_path = simplified_onnx

    # 3. 查找 trtexec
    trtexec = find_trtexec()
    if not trtexec:
        print("❌ 找不到 trtexec，请确认 TensorRT 已正确安装")
        print("   你可以通过 'which trtexec' 查找路径，然后修改脚本中的候选列表")
        return False
    print(f"✅ 找到 trtexec: {trtexec}")

    # 4. 构建命令（使用 --memPoolSize 替代已弃用的 --workspace）
    cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
    cmd.append(f"--memPoolSize=workspace:{workspace_mb}MiB")
    if fp16:
        cmd.append("--fp16")
    if verbose:
        cmd.append("--verbose")
    # 可选：启用所有优化
    # cmd.append("--best")

    print("\n" + "=" * 60)
    print("开始转换 ONNX → TensorRT Engine")
    print("=" * 60)
    print(f"输入 ONNX : {onnx_path}")
    print(f"输出 Engine: {engine_path}")
    print(f"执行命令 : {' '.join(cmd)}")
    print("-" * 60)

    # 5. 执行转换
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            print("\n✅ 转换成功!")
            if os.path.exists(engine_path):
                size_mb = os.path.getsize(engine_path) / 1024 / 1024
                print(f"引擎大小: {size_mb:.2f} MB")
            print("=" * 60)
            return True
        else:
            print("\n❌ 转换失败!")
            print("错误信息:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"\n❌ 执行过程中发生异常: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ONNX to TensorRT Engine with onnx-simplifier")
    parser.add_argument("--onnx", type=str, required=True, help="输入的 ONNX 文件路径")
    parser.add_argument("--engine", type=str, required=True, help="输出的 Engine 文件路径")
    parser.add_argument("--workspace", type=int, default=4096, help="工作空间内存大小 (MiB)，默认 4096")
    parser.add_argument("--no-fp16", action="store_true", help="禁用 FP16（默认启用）")
    parser.add_argument("--no-simplify", action="store_true", help="禁用 ONNX 简化（不推荐）")
    parser.add_argument("--verbose", action="store_true", help="打印 trtexec 详细日志")
    args = parser.parse_args()

    success = onnx_to_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        workspace_mb=args.workspace,
        fp16=not args.no_fp16,
        verbose=args.verbose,
        use_simplify=not args.no_simplify
    )

    if success:
        abs_path = os.path.abspath(args.engine)
        print(f"\n✅ 引擎已生成: {abs_path}")
        print(f"你可以在 params.yaml 中设置:")
        print(f"  engine_path: \"{abs_path}\"")
    else:
        sys.exit(1)