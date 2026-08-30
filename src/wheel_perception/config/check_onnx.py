import cv2
import numpy as np
import onnxruntime as ort
import os

# =========================
# 配置
# =========================
onnx_path = "/home/xuantengxing/xtx-roadseg-rgb/xtx-roadseg_rgb.onnx"
img_path = "/home/xuantengxing/Cityscapes/leftImg8bit/val/munster/munster_000173_000019_leftImg8bit.png"

save_dir = "/home/xuantengxing/xtx-roadseg-rgb/onnx_result"
os.makedirs(save_dir, exist_ok=True)

input_height = 480   # ⚠️ 改成你训练输入尺寸
input_width = 640


# =========================
# 1. 加载 ONNX
# =========================
sess = ort.InferenceSession(
    onnx_path,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

input_name = sess.get_inputs()[0].name
print("ONNX input:", input_name)


# =========================
# 2. 读取并预处理图像
# =========================
img = cv2.imread(img_path)
ori_h, ori_w = img.shape[:2]

img_resized = cv2.resize(img, (input_width, input_height))

# BGR → RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

img_rgb = img_rgb.astype(np.float32) / 255.0

# HWC → CHW
img_chw = np.transpose(img_rgb, (2, 0, 1))

# 增加 batch
input_tensor = np.expand_dims(img_chw, axis=0).astype(np.float32)


# =========================
# 3. ONNX 推理
# =========================
outputs = sess.run(None, {input_name: input_tensor})

seg = outputs[0]   # shape: [1,1,H,W] or [1,2,H,W]

print("Output shape:", seg.shape)


# =========================
# 4. 后处理
# =========================
# 单通道 sigmoid
prob = seg[0, 0]
pred = (prob > 0.5).astype(np.uint8)

# resize 回原图
pred = cv2.resize(pred, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
prob = cv2.resize(prob, (ori_w, ori_h))

# =========================
# 5. 生成绿色掩码
# =========================
green_mask = np.zeros((ori_h, ori_w, 3), dtype=np.uint8)
green_mask[pred == 1] = [0, 255, 0]  # BGR

# =========================
# 6. 叠加
# =========================
overlay = cv2.addWeighted(green_mask, 0.5, img, 1.0, 0)

# =========================
# 7. 保存
# =========================
cv2.imwrite(os.path.join(save_dir, "mask.png"), pred * 255)
cv2.imwrite(os.path.join(save_dir, "overlay.png"), overlay)

print("Saved to:", save_dir)