#!/bin/bash
# =============================================================
# build_all_trt10.sh — 整体切换到 TensorRT 10 的一键脚本
#
# 功能：
#   1. 检查/提示安装 CUDA 12 运行库（pip 方案 B）
#   2. 编译 build_engine_trt10（C++ 构建器，替代 trtexec）
#   3. 转换 ESS 引擎: light_ess.engine + ess.engine（带插件）
#   4. 转换 AI 引擎:  xtx-roadseg_rgb.onnx → combined4.engine
#   5. 更新 CMakeLists.txt 的 TRT_ROOT 指向 TRT 10.9（自动备份）
#
# 前提：本机已有 TRT 10.9 库（/usr/local/cuda-12.2/targets/x86_64-linux）
# 用法：bash build_all_trt10.sh
# =============================================================
set -euo pipefail

# ---------- 路径配置 ----------
ESS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$ESS_DIR")"                       # .../src/wheel_perception
SRC_DIR="$(dirname "$PKG_DIR")"                       # .../src
WS_DIR="$(dirname "$SRC_DIR")"                        # .../wheel_ess
CMAKELISTS="$PKG_DIR/CMakeLists.txt"

TRT10_ROOT="/usr/local/cuda-12.2/targets/x86_64-linux"   # 本机 TRT 10.9（库+头文件）
TRT10_INC="$TRT10_ROOT/include"
TRT10_LIB="$TRT10_ROOT/lib"
PY_SITE="$(python3 -c 'import site; print(site.USER_SITE)')"
NV_DIR="$PY_SITE/nvidia"

# PyTorch 环境（pth → onnx 导出用，本机在 miniconda trt_env）
PY_TRT="/home/smy/miniconda3/envs/trt_env/bin/python3"

ESS_PLUGIN="$ESS_DIR/plugins/x86_64/ess_plugins.so"
ESS_INPUT_SHAPES="input_left:1x3x288x480;input_right:1x3x288x480"
AI_ENGINE="$PKG_DIR/config/combined4.engine"

echo "=============================================="
echo " wheel_ess → TensorRT 10 一键切换"
echo "=============================================="
echo " TRT10 (库+头文件): $TRT10_ROOT"
echo " pip CUDA12 目录 : $NV_DIR"
echo " 包目录           : $PKG_DIR"

# ---------- 0. 检查 pip CUDA 12 运行库 ----------
check_dir() {
    if [ ! -d "$1" ]; then
        echo ""
        echo "❌ 缺少: $1"
        echo ""
        echo "请先安装 CUDA 12 运行库（方案 B）:"
        echo "  pip install --user nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \\"
        echo "               nvidia-cusparse-cu12 nvidia-cudnn-cu12"
        echo ""
        return 1
    fi
    echo " ✅ $1"
}
check_dir "$NV_DIR/cuda_runtime/lib"
check_dir "$NV_DIR/cublas/lib"
check_dir "$NV_DIR/cusparse/lib"
check_dir "$NV_DIR/cudnn/lib"

if [ ! -d "$TRT10_LIB" ] || [ ! -f "$TRT10_LIB/libnvinfer.so.10" ]; then
    echo "❌ 本机缺少 TRT 10 库: $TRT10_LIB/libnvinfer.so.10"
    exit 1
fi
echo " ✅ TRT 10 库存在 ($TRT10_LIB)"

export LD_LIBRARY_PATH="$TRT10_LIB:$NV_DIR/cuda_runtime/lib:$NV_DIR/cublas/lib:$NV_DIR/cusparse/lib:$NV_DIR/cudnn/lib:${LD_LIBRARY_PATH:-}"

# ---------- 1. 编译构建器 ----------
echo ""
echo "── [1/4] 编译 build_engine_trt10 ..."
if [ ! -f "$TRT10_INC/NvInfer.h" ]; then
    echo "❌ 缺少 TRT 头文件: $TRT10_INC/NvInfer.h"
    exit 1
fi
g++ -std=c++17 -O2 "$ESS_DIR/build_engine_trt10.cpp" \
    -I"$TRT10_INC" -I/usr/local/cuda-11.8/include \
    -L"$TRT10_LIB" -lnvinfer -lnvonnxparser -ldl \
    -o "$ESS_DIR/build_engine_trt10"
echo " ✅ 构建器编译完成: $ESS_DIR/build_engine_trt10"

BUILDER="$ESS_DIR/build_engine_trt10"

# ---------- 2. 转换 ESS 引擎（带插件） ----------
build_onnx() {
    local onnx="$1" engine="$2" extra="${3:-}"
    if [ -f "$engine" ]; then
        cp "$engine" "$engine.bak-$(date +%Y%m%d%H%M%S)" 2>/dev/null || true
    fi
    echo ""
    echo "── [2/4] 转换 $onnx → $engine ..."
    "$BUILDER" --onnx="$onnx" --saveEngine="$engine" \
        --plugin="$ESS_PLUGIN" --fp16 --input-shapes="$ESS_INPUT_SHAPES"
    echo " ✅ $engine 生成完毕 ($(stat -c%s "$engine") bytes)"
}

build_onnx "$ESS_DIR/light_ess.onnx" "$ESS_DIR/light_ess.engine"
build_onnx "$ESS_DIR/ess.onnx" "$ESS_DIR/ess.engine"

# ---------- 3. 转换 AI 引擎（pth → onnx → engine，走 pth2engine.py） ----------
echo ""
echo "── [3/4] 转换 AI 引擎（pth2engine.py: bisenet_combined4.pth → combined4.engine）..."
PTH2ENGINE="$PKG_DIR/config/pth2engine.py"
if [ ! -f "$PTH2ENGINE" ]; then
    echo "❌ 缺少 $PTH2ENGINE"
    exit 1
fi
if [ ! -x "$PY_TRT" ]; then
    echo "❌ 缺少 PyTorch 环境，未找到 $PY_TRT"
    echo "   请确认 miniconda trt_env 存在: /home/smy/miniconda3/envs/trt_env"
    exit 1
fi
if ! "$PY_TRT" -c "import torch" 2>/dev/null; then
    echo "❌ $PY_TRT 中缺少 PyTorch:"
    echo "   $PY_TRT -m pip install torch torchvision"
    exit 1
fi
if [ -f "$AI_ENGINE" ]; then
    cp "$AI_ENGINE" "$AI_ENGINE.bak-$(date +%Y%m%d%H%M%S)"
    echo "   旧引擎已备份: $AI_ENGINE.bak-*"
fi
cd "$PKG_DIR/config" && "$PY_TRT" "$PTH2ENGINE" || exit 1
echo " ✅ $AI_ENGINE 生成完毕 ($(stat -c%s "$AI_ENGINE") bytes)"

# ---------- 4. 更新 CMakeLists ----------
echo ""
echo "── [4/4] 更新 CMakeLists.txt 指向 TRT 10 ..."
cp "$CMAKELISTS" "$CMAKELISTS.bak-$(date +%Y%m%d%H%M%S)"
python3 - "$CMAKELISTS" "$TRT10_ROOT" <<'PYEOF'
import sys
path, trt_root = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as f:
    text = f.read()

# 1) TRT_ROOT 指向 TRT 10.9
old_root = 'set(TRT_ROOT "/home/smy/wheel_cuda/TensorRT-8.6.1.6")'
new_root = f'set(TRT_ROOT "{trt_root}")  # TensorRT 10.9（CUDA 12 运行库）'
if old_root not in text:
    print("⚠️ 未找到原 TRT_ROOT 行，请手动检查 CMakeLists.txt")
else:
    text = text.replace(old_root, new_root)

# 2) include_directories 增加 TRT 10 头文件路径（避免误用旧 NvInfer.h）
if '${TRT_ROOT}/include' not in text:
    text = text.replace(
        '  ${OpenCV_INCLUDE_DIRS}\n)',
        '  ${OpenCV_INCLUDE_DIRS}\n  ${TRT_ROOT}/include   # TensorRT 10 头文件\n)',
        1,
    )

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print(" ✅ CMakeLists.txt 已更新（备份: *.bak-*）")
PYEOF

# ---------- 总结 ----------
echo ""
echo "=============================================="
echo " 全部完成！接下来："
echo ""
echo " 1. 重新编译（此时用 TRT 10 头文件+库）:"
echo "    cd $WS_DIR"
echo "    source /opt/ros/humble/setup.bash"
echo "    colcon build --allow-overriding cv_bridge image_geometry"
echo ""
echo " 2. 运行时加载 TRT10 + CUDA12 环境:"
echo "    source $ESS_DIR/trt10_env.sh"
echo "    (或每次运行前手动设置 LD_LIBRARY_PATH)"
echo ""
echo " 3. 如需回退到 TRT 8.6:"
echo "    cp $CMAKELISTS.bak-* $CMAKELISTS  # 并重新 colcon build"
echo "=============================================="
