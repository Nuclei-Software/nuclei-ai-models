# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# See https://github.com/triple-Mu/yolov8/blob/main/examples/YOLOv8-TFLite-Python/main.py

import argparse
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import yaml
from typing import List, Dict, Tuple, Optional
import os

# from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

def cxcywh_to_xyxy(x: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from cxcywh format to xyxy format.

    Args:
        x (np.ndarray): Bounding boxes in cxcywh format.

    Returns:
        np.ndarray: Bounding boxes in xyxy format.
    """
    if len(x) > 0:
        # Convert center x, center y, width, height to x_min, y_min, x_max, y_max
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]
    return x

def custom_draw(image: np.ndarray, boxes: List[Tuple[float, float, float, float, float, float]], masks: List[np.ndarray],
                color_palette: callable, prediction_result_dir: str, file: str, class_names: Optional[dict] = None) -> None:
    """
    Draws bounding boxes and masks on an image and saves the result.

    Args:
        image (np.ndarray): The input image.
        boxes (List[Tuple[float, float, float, float, float, float]]): List of bounding boxes with confidence and class.
        masks (List[np.ndarray]): List of masks corresponding to the bounding boxes.
        color_palette (callable): Function to get color for a class.
        prediction_result_dir (str): Directory to save the result image.
        file (str): File name for the result image.
        class_names (Optional[dict]): File containing class names. If None, use class indices.

    Returns:
        None
    """

    overlay_image = image.copy()
    for (*box, conf, cls_), mask in zip(boxes, masks):
        # color = color_palette(int(cls_), as_bgr=True)
        color = color_palette
        class_label = class_names[int(cls_)] if class_names else str(int(cls_))
        for c in range(3):
            overlay_image[:, :, c] = np.where(mask == 1, overlay_image[:, :, c] * 0.5, overlay_image[:, :, c])
        cv2.rectangle(overlay_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (124, 211, 32, 0.5), 1, cv2.LINE_AA)
        cv2.putText(overlay_image, f"{class_label}: {conf:.3f}", (int(box[0]), int(box[1] - 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (124, 211, 32, 0.5), 2, cv2.LINE_AA)
    
    overlay_image = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)
    return overlay_image

def multiply_tensors(masks_in: np.ndarray, reshaped_protos: np.ndarray) -> np.ndarray:
    """
    Multiplies two tensors manually.

    Args:
        masks_in (np.ndarray): First tensor.
        reshaped_protos (np.ndarray): Second tensor.

    Returns:
        np.ndarray: Result of the multiplication.
    """
    # Initialize the result tensor with zeros
    result = np.zeros((masks_in.shape[0], reshaped_protos.shape[1]))

    # Perform matrix multiplication manually
    for i in range(masks_in.shape[0]):  # Iterate over rows of masks_in
        for j in range(reshaped_protos.shape[1]):  # Iterate over columns of reshaped_protos
            for k in range(masks_in.shape[1]):  # Iterate over columns of masks_in / rows of reshaped_protos
                result[i, j] += masks_in[i, k] * reshaped_protos[k, j]

    return result

class YOLOv8TFLite:
    """
    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    This class handles model loading, preprocessing, inference, and visualization of detection results.

    Attributes:
        model (Interpreter): TensorFlow Lite interpreter for the YOLOv8 model.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        classes (Dict[int, str]): Dictionary mapping class IDs to class names.
        color_palette (np.ndarray): Random color palette for visualization with shape (num_classes, 3).
        in_width (int): Input width required by the model.
        in_height (int): Input height required by the model.
        in_index (int): Input tensor index in the model.
        in_scale (float): Input quantization scale factor.
        in_zero_point (int): Input quantization zero point.
        int8 (bool): Whether the model uses int8 quantization.
        out_index (int): Output tensor index in the model.
        out_scale (float): Output quantization scale factor.
        out_zero_point (int): Output quantization zero point.

    Methods:
        letterbox: Resizes and pads image while maintaining aspect ratio.
        draw_detections: Draws bounding boxes and labels on the input image.
        preprocess: Preprocesses the input image before inference.
        postprocess: Processes model outputs to extract and visualize detections.
        detect: Performs object detection on an input image.
    """

    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.45, metadata: Union[str, None] = None):
        """
        Initialize an instance of the YOLOv8TFLite class.

        Args:
            model (str): Path to the TFLite model file.
            conf (float): Confidence threshold for filtering detections.
            iou (float): IoU threshold for non-maximum suppression.
            metadata (str | None): Path to the metadata file containing class names.
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(42)  # Set seed for reproducible colors
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        # Initialize the TFLite interpreter
        self.model = Interpreter(model_path=model)
        self.model.allocate_tensors()

        # Get input details
        input_details = self.model.get_input_details()[0]
        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        # Get output details
        detections_details = self.model.get_output_details()[0]
        self.detections_index = detections_details["index"]
        self.detections_scale, self.detections_zero_point = detections_details["quantization"]

        masks_details = self.model.get_output_details()[1]
        self.masks_index = masks_details["index"]
        self.masks_scale, self.masks_zero_point = masks_details["quantization"]

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int) -> None:
        """
        Draw bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (np.ndarray): Detected bounding box in the format [x1, y1, width, height].
            score (np.float32): Confidence score of the detection.
            class_id (int): Class ID for the detected object.
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        # Create label with class name and score
        label = f"{self.classes[class_id]}: {score:.2f}"

        print(f"Position is ({int(x1)}, {int(y1)}, {int(x1 + w)}, {int(y1 + h)}), Score is %.2f"%(score))

        # Get text size for background rectangle
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        # Position label above or below box depending on space
        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        # Draw label background
        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        # Draw text
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocess the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed with shape (H, W, C).

        Returns:
            (np.ndarray): Preprocessed image ready for model input.
            (Tuple[float, float]): ratios for coordinate adjustment.
        """
        origin_shape = img.shape[:2]
        img = cv2.resize(img, (self.in_width, self.in_height), interpolation=cv2.INTER_LINEAR)
        new_shape = img.shape[:2]
        img = img[..., ::-1][None]  # N,H,W,C for TFLite
        img = np.ascontiguousarray(img)
        img = img.astype(np.uint8)

        print(f"Origin image shape is: {origin_shape}")
        print(f"Resize image shape is: {new_shape}")

        ratio = (origin_shape[0] / new_shape[0], origin_shape[1] / new_shape[1])
        return img, ratio

    def postprocess(self, image: np.ndarray, masks: np.ndarray, detections: np.ndarray, 
                conf_threshold: float, iou_threshold: float, n_masks: int, width: int, height: int, 
                prediction_result_dir: str, file: str, class_names: dict) -> None:
        """
        Post-process the predictions and save the results.

        Args:
            image (np.ndarray): The original image.
            masks (np.ndarray): The predicted masks.
            detections (np.ndarray): The predicted detections.
            conf_threshold (float): Confidence threshold.
            iou_threshold (float): IoU threshold.
            n_masks (int): Number of masks.
            width (int): Width of the image.
            height (int): Height of the image.
            prediction_result_dir (str): Directory to save the result image.
            file (str): File name for the result image.

        Returns:
            None
        """

        detections_t = np.transpose(detections, (0, 2, 1))
        masks_deq = masks

    
        # Filter detections by score
        detections = detections_t[np.amax(detections_t[..., 4:-n_masks], axis=-1) > conf_threshold]
        # Scale normalized box to model width and height
        detections[..., [0, 2]] *= width
        detections[..., [1, 3]] *= height

        # Convert raw detections to final detections structure
        detections = np.c_[detections[..., :4], np.amax(detections[..., 4:-n_masks], axis=-1),
                           np.argmax(detections[..., 4:-n_masks], axis=-1), detections[..., -n_masks:]]
    
        # Apply NMS
        nmsed_detections = detections[cv2.dnn.NMSBoxes(detections[:, :4].tolist(), detections[:, 4].tolist(), 
                                                       conf_threshold, iou_threshold)]

        if nmsed_detections.shape[0] > 0:
            # Transpose masks
            masks_t = np.transpose(masks_deq, (0, 3, 1, 2))

            # Flatten the masks
            squeezed_masks = np.squeeze(masks_t)
            flattened_masks = squeezed_masks.reshape((squeezed_masks.shape[0], -1))

            # Matrix multiplication between detection masks and mask_buffer
            detections_mask = nmsed_detections[:, 6:]
            post_processed_masks = multiply_tensors(detections_mask, flattened_masks)

            # Restore masks initial shape
            b, c, mh, mw = masks_t.shape
            post_processed_masks = post_processed_masks.reshape((-1, mh, mw))

            # Make the masks binary
            binary_masks = (post_processed_masks > 0.5).astype("uint8")

            # Make the masks channel last to scale them up
            binary_masks_t = np.transpose(binary_masks, (1, 2, 0))

            # Scale binary masks to the initial image size
            scaled_masks = cv2.resize(binary_masks_t, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

            # Convert the masks back to channel first
            if nmsed_detections.shape[0] == 1:
                scaled_masks = np.expand_dims(scaled_masks, -1)
            scaled_masks_t = np.transpose(scaled_masks, (2, 0, 1))  # (520, 520, 3) => (3, 520, 520)

            # Normalize then scale the boxes to the initial image size
            nmsed_detections[..., [0, 2]] /= width
            nmsed_detections[..., [1, 3]] /= height
            nmsed_detections[..., [0, 2]] *= image.shape[1]
            nmsed_detections[..., [1, 3]] *= image.shape[0]

            # Bounding boxes format change: cxcywh -> xyxy
            xyxy_detections = cxcywh_to_xyxy(nmsed_detections[..., 0:6])
            np.random.seed(42)  # Set seed for reproducible colors
            colors = np.random.uniform(128, 255, size=(80, 3))
            return custom_draw(image, xyxy_detections, scaled_masks_t, colors, prediction_result_dir, file, class_names)
        else:
            print(f"No detections found in image {file}")

    def detect(self, img_path: str) -> np.ndarray:
        """
        Perform object detection on an input image.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        # Load and preprocess image
        img = cv2.imread(img_path)
        x, ratio = self.preprocess(img)

        # Apply quantization if model is int8
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Set input tensor and run inference
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        # Get output and dequantize if necessary
        detections = self.model.get_tensor(self.detections_index)
        masks = self.model.get_tensor(self.masks_index)
        if self.int8:
            detections = (detections.astype(np.float32) - self.detections_zero_point) * self.detections_scale
            masks = (masks.astype(np.float32) - self.masks_zero_point) * self.masks_scale

        # Process detections and return result
        n_masks = 32

        prediction_result_dir = "./"
        return self.postprocess(img, masks, detections, self.conf, self.iou, n_masks,
                    self.in_width, self.in_height, prediction_result_dir, img_path, self.classes)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="../pretrained_models/yolov8n_256_quant_pc_ii_seg_coco-st.tflite",
        help="Path to TFLite model.",
    )
    parser.add_argument("--img", type=str, default=str("../../../../public_dataset/COCO_person/000000000036.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.65, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--metadata", type=str, default="../../../../public_dataset/COCO_person/coco80.yaml", help="Metadata yaml")
    args = parser.parse_args()
    detector = YOLOv8TFLite(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(args.img)
    print("Show in result.jpg")
    cv2.imwrite("result.jpg", result)
    # cv2.imshow("Output", result)
    # cv2.waitKey(0)
