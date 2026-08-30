#!/bin/bash
# =============================================================
# trt10_env.sh — 运行时加载 TRT 10.9 + CUDA 12 环境
#
# 用法:  source trt10_env.sh    （在运行节点前执行）
# 之后即可正常运行: ros2 launch ... / 直接运行可执行文件
#
# 说明:
#   - 不修改系统任何文件，只在当前 shell 临时追加 LD_LIBRARY_PATH
#   - CUDA 12 运行库来自 pip 安装的 nvidia-*-cu12 包（~/.local 下）
#   - TRT 10.9 库来自 /usr/local/cuda-12.2/targets/x86_64-linux
#   - 系统 CUDA 11.8 不受影响（程序链接的 libcudart.so.11 仍用系统库）
# =============================================================
set -euo pipefail

TRT10_LIB="/usr/local/cuda-12.2/targets/x86_64-linux/lib"
PY_SITE="$(python3 -c 'import site; print(site.USER_SITE)')"
NV_DIR="$PY_SITE/nvidia"

for d in "$TRT10_LIB" \
         "$NV_DIR/cuda_runtime/lib" \
         "$NV_DIR/cublas/lib" \
         "$NV_DIR/cusparse/lib" \
         "$NV_DIR/cudnn/lib"; do
    if [ ! -d "$d" ]; then
        echo "[trt10_env] ❌ 缺少目录: $d"
        echo "[trt10_env] 请先执行: pip install --user nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 nvidia-cusparse-cu12 nvidia-cudnn-cu12"
        return 1 2>/dev/null || exit 1
    fi
    export LD_LIBRARY_PATH="$d:$LD_LIBRARY_PATH"
done

echo "[trt10_env] ✅ TRT 10.9 + CUDA 12 运行库已加载"
echo "[trt10_env]    LD_LIBRARY_PATH 已包含:"
echo "[trt10_env]      $TRT10_LIB"
echo "[trt10_env]      $NV_DIR/cuda_runtime/lib"
echo "[trt10_env]      $NV_DIR/cublas/lib"
echo "[trt10_env]      $NV_DIR/cusparse/lib"
echo "[trt10_env]      $NV_DIR/cudnn/lib"
