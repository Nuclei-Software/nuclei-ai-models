import numpy as np
import cv2
import tensorflow as tf

# -------------------------------
# 配置部分
# -------------------------------
MODEL_PATH = "../pretrained_models/COCO_Person/movenet_lightning_heatmaps_192/movenet_lightning_heatmaps_192_int8_pc.tflite"            # 替换为你的 .tflite 模型路径
IMAGE_PATH = "../../../../public_dataset/movenet/0000001.jpg"                 # 替换为你要测试的图像路径
INPUT_HEIGHT = 192                      # 根据模型输入尺寸调整
INPUT_WIDTH = 192                       # 根据模型输入尺寸调整
NUM_KEYPOINTS = 17                      # 关键点数量 K

# -------------------------------
# 步骤 1: 加载 TFLite 模型
# -------------------------------
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input details:", input_details)
print("Output details:", output_details)

# -------------------------------
# 步骤 2: 读取并预处理图像
# -------------------------------
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError(f"无法读取图像: {IMAGE_PATH}")

# OpenCV 默认是 BGR，转成 RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 调整大小到模型输入尺寸
resized_img = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT))

# 扩展 batch 维度，并转换为 UINT8 类型
input_data = np.expand_dims(resized_img, axis=0).astype(input_details['dtype'])

# -------------------------------
# 步骤 3: 设置输入并运行推理
# -------------------------------
interpreter.set_tensor(input_details['index'], input_data)
interpreter.invoke()

# -------------------------------
# 步骤 4: 获取输出热图
# -------------------------------
output_data = interpreter.get_tensor(output_details['index'])

# 输出数据类型是 FLOAT32
print("Heatmap shape:", output_data.shape)  # 应该是 (1, W, H, K)

# -------------------------------
# 可选：可视化热图或提取关键点
# -------------------------------

def visualize_heatmap(heatmap, image, keypoint_idx=0):
    """
    可视化第 keypoint_idx 个关键点的热图
    """
    import matplotlib.pyplot as plt

    # 热图为 float，先归一化到 0-1
    heatmap_single = heatmap[0, :, :, keypoint_idx]
    heatmap_resized = cv2.resize(heatmap_single, (image.shape[1], image.shape[0]))
    heatmap_normalized = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min())

    plt.imshow(image)
    plt.imshow(heatmap_normalized, alpha=0.5, cmap='jet')
    plt.colorbar()
    plt.title(f"Heatmap for Keypoint {keypoint_idx}")
    plt.axis('off')
    plt.show()

# 示例：可视化第一个关键点的热图
visualize_heatmap(output_data, img)

# -------------------------------
# 可选：从热图中提取关键点坐标
# -------------------------------

def get_keypoints_from_heatmap(heatmap_batch, threshold=0.1):
    """
    从热图中提取关键点坐标
    heatmap_batch: 形状为 (1, W, H, K)
    返回: list of (x, y) coordinates for each keypoint
    """
    _, W, H, K = heatmap_batch.shape
    keypoints = []

    for k in range(K):
        hm = heatmap_batch[0, :, :, k]
        max_val = hm.max()
        if max_val < threshold:
            keypoints.append((None, None))  # 置信度过低
        else:
            y, x = np.unravel_index(hm.argmax(), hm.shape)
            # 映射回原始图像分辨率（假设原始图像为原图）
            orig_x = int(x * img.shape[1] / W)
            orig_y = int(y * img.shape[0] / H)
            keypoints.append((orig_x, orig_y))
    return keypoints

keypoints = get_keypoints_from_heatmap(output_data)
for i, (x, y) in enumerate(keypoints):
    print(f"Keypoint {i}: ({x}, {y})")
    if x is not None and y is not None:
        cv2.circle(img, (x, y), radius=5, color=(0, 255, 0), thickness=-1)

print("Show in result.jpg")
cv2.imwrite("result_headmap.jpg", img)

# cv2.imshow("Detected Keypoints", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()