#!/usr/bin/env python3
import argparse
import ctypes
import importlib
import os
import sys
import time
from dataclasses import dataclass
from typing import Tuple, Optional

import cv2
import numpy as np

TRT_ROOT = "/home/smy/alpha_ws/TensorRT-8.6.1.6"
TRT_PYTHON_DIR = os.path.join(TRT_ROOT, "python")
DEFAULT_ENGINE = os.path.join(TRT_PYTHON_DIR, "bisenetv2_model.engine")
DEFAULT_VIDEO = "/home/smy/alpha_ws/haizhuhu.mp4"
ROAD_CLASS_ID = 0
MEAN = np.array([0.3257, 0.3690, 0.3223], dtype=np.float32)
STD = np.array([0.2112, 0.2148, 0.2115], dtype=np.float32)


def ensure_tensorrt_importable() -> None:
    if TRT_PYTHON_DIR not in sys.path:
        sys.path.insert(0, TRT_PYTHON_DIR)

    try:
        importlib.import_module("tensorrt")
        return
    except ImportError:
        pass

    major = sys.version_info.major
    minor = sys.version_info.minor
    wheel_name = f"tensorrt-8.6.1-cp{major}{minor}-none-linux_x86_64.whl"
    wheel_path = os.path.join(TRT_PYTHON_DIR, wheel_name)
    raise ImportError(
        "无法导入 tensorrt。请先安装本地 wheel，例如：\n"
        f"pip install {wheel_path}"
    )


ensure_tensorrt_importable()
trt = importlib.import_module("tensorrt")


class TRTLogger(trt.ILogger):
    def __init__(self) -> None:
        super().__init__()

    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.WARNING:
            print(f"[TensorRT] {msg}")


class CudaError(RuntimeError):
    pass


class CudaRuntime:
    def __init__(self) -> None:
        self.lib = self._load_libcudart()
        self.cudaMemcpyHostToDevice = 1
        self.cudaMemcpyDeviceToHost = 2
        self._configure_signatures()

    @staticmethod
    def _load_libcudart() -> ctypes.CDLL:
        candidates = [
            "libcudart.so",
            "/usr/local/cuda/lib64/libcudart.so",
            "/usr/local/cuda-11.8/lib64/libcudart.so",
        ]
        for path in candidates:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
        raise RuntimeError("找不到 libcudart.so，请确认 CUDA Runtime 已安装")

    def _configure_signatures(self) -> None:
        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int

        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int

        self.lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.cudaMemcpy.restype = ctypes.c_int

        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int

        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int

    def check(self, code: int, name: str) -> None:
        if code != 0:
            raise CudaError(f"{name} 失败，CUDA 错误码: {code}")

    def malloc(self, nbytes: int) -> int:
        ptr = ctypes.c_void_p()
        self.check(self.lib.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
        return ptr.value

    def free(self, ptr: int) -> None:
        if ptr:
            self.check(self.lib.cudaFree(ctypes.c_void_p(ptr)), "cudaFree")

    def memcpy_htod(self, dst: int, src: np.ndarray) -> None:
        contiguous = np.ascontiguousarray(src)
        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(dst),
                ctypes.c_void_p(contiguous.ctypes.data),
                contiguous.nbytes,
                self.cudaMemcpyHostToDevice,
            ),
            "cudaMemcpy(HtoD)",
        )

    def memcpy_dtoh(self, dst: np.ndarray, src: int) -> None:
        contiguous = np.ascontiguousarray(dst)
        self.check(
            self.lib.cudaMemcpy(
                ctypes.c_void_p(contiguous.ctypes.data),
                ctypes.c_void_p(src),
                contiguous.nbytes,
                self.cudaMemcpyDeviceToHost,
            ),
            "cudaMemcpy(DtoH)",
        )
        np.copyto(dst, contiguous)

    def create_stream(self) -> int:
        stream = ctypes.c_void_p()
        self.check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream.value

    def synchronize_stream(self, stream: int) -> None:
        self.check(self.lib.cudaStreamSynchronize(ctypes.c_void_p(stream)), "cudaStreamSynchronize")

    def destroy_stream(self, stream: int) -> None:
        if stream:
            self.check(self.lib.cudaStreamDestroy(ctypes.c_void_p(stream)), "cudaStreamDestroy")


@dataclass
class Binding:
    name: str
    dtype: np.dtype
    shape: tuple
    nbytes: int
    ptr: int
    is_input: bool


class TRTBiSeNetVideoInfer:
    def __init__(self, engine_path: str) -> None:
        self.engine_path = engine_path
        self.logger = TRTLogger()
        self.cuda = CudaRuntime()
        self.stream = self.cuda.create_stream()
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self._allocate_bindings()
        self.input_binding = next(binding for binding in self.bindings if binding.is_input)
        self.output_binding = next(binding for binding in self.bindings if not binding.is_input)
        self.input_h = self.input_binding.shape[-2]
        self.input_w = self.input_binding.shape[-1]
        self.output_host = np.empty(self.output_binding.shape, dtype=self.output_binding.dtype)

    def _load_engine(self, engine_path: str):
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"找不到 engine 文件: {engine_path}")
        with open(engine_path, "rb") as f:
            engine = self.runtime.deserialize_cuda_engine(f.read())
        if engine is None:
            raise RuntimeError("TensorRT engine 反序列化失败")
        return engine

    def _tensor_name(self, index: int) -> str:
        if hasattr(self.engine, "get_tensor_name"):
            return self.engine.get_tensor_name(index)
        return self.engine.get_binding_name(index)

    def _tensor_shape(self, name: str, index: int) -> tuple:
        if hasattr(self.engine, "get_tensor_shape"):
            shape = tuple(self.engine.get_tensor_shape(name))
        else:
            shape = tuple(self.engine.get_binding_dimensions(index))
        shape = tuple(1 if dim < 0 else dim for dim in shape)
        return shape

    def _tensor_dtype(self, name: str, index: int):
        if hasattr(self.engine, "get_tensor_dtype"):
            return trt.nptype(self.engine.get_tensor_dtype(name))
        return trt.nptype(self.engine.get_binding_dtype(index))

    def _tensor_is_input(self, name: str, index: int) -> bool:
        if hasattr(self.engine, "get_tensor_mode"):
            return self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
        return self.engine.binding_is_input(index)

    def _set_input_shape_if_needed(self, name: str, shape: tuple) -> None:
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(name, shape)
        elif hasattr(self.context, "set_binding_shape"):
            input_index = 0
            for idx, binding in enumerate(self.bindings):
                if binding.name == name:
                    input_index = idx
                    break
            self.context.set_binding_shape(input_index, shape)

    def _allocate_bindings(self):
        bindings = []
        num_tensors = self.engine.num_io_tensors if hasattr(self.engine, "num_io_tensors") else self.engine.num_bindings
        for index in range(num_tensors):
            name = self._tensor_name(index)
            shape = self._tensor_shape(name, index)
            dtype = np.dtype(self._tensor_dtype(name, index))
            is_input = self._tensor_is_input(name, index)
            if is_input:
                self._set_input_shape_if_needed(name, shape)
            nbytes = int(np.prod(shape)) * dtype.itemsize
            ptr = self.cuda.malloc(nbytes)
            bindings.append(Binding(name=name, dtype=dtype, shape=shape, nbytes=nbytes, ptr=ptr, is_input=is_input))
        return bindings

    def preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame_bgr, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - MEAN) / STD
        nchw = np.transpose(normalized, (2, 0, 1))[None, ...]
        return np.ascontiguousarray(nchw.astype(self.input_binding.dtype))

    def infer(self, frame_bgr: np.ndarray) -> np.ndarray:
        input_tensor = self.preprocess(frame_bgr)
        self.cuda.memcpy_htod(self.input_binding.ptr, input_tensor)

        if hasattr(self.context, "set_tensor_address"):
            for binding in self.bindings:
                self.context.set_tensor_address(binding.name, binding.ptr)
            ok = self.context.execute_async_v3(stream_handle=self.stream)
        else:
            ptrs = [binding.ptr for binding in self.bindings]
            ok = self.context.execute_async_v2(bindings=ptrs, stream_handle=self.stream)

        if not ok:
            raise RuntimeError("TensorRT 推理执行失败")

        self.cuda.synchronize_stream(self.stream)
        self.cuda.memcpy_dtoh(self.output_host, self.output_binding.ptr)
        logits = self.output_host
        if logits.ndim != 4:
            raise RuntimeError(f"输出维度异常: {logits.shape}")
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
        return mask

    def overlay_road_mask(self, frame_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> Tuple[np.ndarray, np.ndarray]:
        road_mask = (mask == ROAD_CLASS_ID).astype(np.uint8) * 255
        road_mask = cv2.resize(road_mask, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = frame_bgr.copy()
        overlay[road_mask > 0] = (0, 255, 0)
        blended = cv2.addWeighted(frame_bgr, 1.0 - alpha, overlay, alpha, 0.0)
        result = frame_bgr.copy()
        result[road_mask > 0] = blended[road_mask > 0]
        return result, road_mask

    def release(self) -> None:
        for binding in self.bindings:
            self.cuda.free(binding.ptr)
        self.cuda.destroy_stream(self.stream)


def build_output_paths(video_path: str, output_dir: Optional[str]) -> Tuple[str, str]:
    if output_dir is None:
        output_dir = os.path.dirname(video_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    overlay_path = os.path.join(output_dir, f"{stem}_road_overlay.mp4")
    mask_path = os.path.join(output_dir, f"{stem}_road_mask.mp4")
    return overlay_path, mask_path


def main() -> None:
    parser = argparse.ArgumentParser(description="使用 TensorRT BiSeNetV2 对视频做道路分割推理")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="TensorRT engine 路径")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="输入视频路径")
    parser.add_argument("--output-dir", default=None, help="输出目录，只有 --save-video 时才会用到")
    parser.add_argument("--display", action="store_true", default=True, help="实时显示结果")
    parser.add_argument("--save-video", action="store_true", help="保存 overlay 和 mask 视频")
    parser.add_argument("--no-loop", action="store_true", help="视频结束后不循环播放")
    parser.add_argument("--alpha", type=float, default=0.35, help="道路 mask 透明度")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        raise FileNotFoundError(f"找不到视频文件: {args.video}")

    inferencer = TRTBiSeNetVideoInfer(args.engine)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        inferencer.release()
        raise RuntimeError(f"无法打开视频: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    overlay_path = None
    mask_path = None
    overlay_writer = None
    mask_writer = None

    if args.save_video:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        overlay_path, mask_path = build_output_paths(args.video, args.output_dir)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        overlay_writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))
        mask_writer = cv2.VideoWriter(mask_path, fourcc, fps, (width, height), isColor=False)

    frame_count = 0
    start = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.no_loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            mask = inferencer.infer(frame)
            overlay_frame, road_mask = inferencer.overlay_road_mask(frame, mask, alpha=args.alpha)

            if overlay_writer is not None and mask_writer is not None:
                overlay_writer.write(overlay_frame)
                mask_writer.write(road_mask)

            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start
                avg_fps = frame_count / max(elapsed, 1e-6)
                print(f"已处理 {frame_count} 帧，平均推理速度: {avg_fps:.2f} FPS")

            if args.display:
                cv2.imshow("road_overlay", overlay_frame)
                cv2.imshow("road_mask", road_mask)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()
        if mask_writer is not None:
            mask_writer.release()
        inferencer.release()
        if args.display:
            cv2.destroyAllWindows()

    total_time = time.time() - start
    avg_fps = frame_count / max(total_time, 1e-6)
    print("处理完成")
    print(f"输入视频: {args.video}")
    if overlay_path is not None and mask_path is not None:
        print(f"overlay 输出: {overlay_path}")
        print(f"mask 输出: {mask_path}")
    print(f"总帧数: {frame_count}")
    print(f"平均 FPS: {avg_fps:.2f}")


if __name__ == "__main__":
    main()
