# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# See https://github.com/triple-Mu/yolov8/blob/main/examples/YOLOv8-TFLite-Python/main.py

import argparse
from typing import Tuple, Union
import sys
import cv2
import numpy as np
import tensorflow as tf
import yaml

# from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class TinyYOLOv2TFLite:
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

    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.25, metadata: Union[str, None] = None):
        """
        Initialize an instance of the TinyYOLOv2TFLite class.

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
        np.random.seed(55)  # Set seed for reproducible colors
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
        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

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

        shape = img.shape[:2]
        x2 = min(x1 + w, shape[1] - 1)
        y2 = min(y1 + h, shape[0] - 1)
        color = self.color_palette[class_id]
        # Draw bounding box
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

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

    def postprocess(self, img: np.ndarray, outputs: np.ndarray, ratio: Tuple[float, float]) -> np.ndarray:
        """
        Process model outputs to extract and visualize detections.

        Args:
            img (np.ndarray): The original input image.
            outputs (np.ndarray): Raw model outputs.
            pad (Tuple[float, float]): Padding ratios from preprocessing.

        Returns:
            (np.ndarray): The input image with detections drawn on it.
        """
        # Adjust coordinates based on padding and scale to original image size
        _,ny,nx,_ = outputs.shape
        # print(outputs)
        anchors = np.zeros([5, 1, 1, 2], dtype=np.float32)
        anchors[0,0,0,:] = [0.076023, 0.258508]
        anchors[1,0,0,:] = [0.163031, 0.413531]
        anchors[2,0,0,:] = [0.234769, 0.702585]
        anchors[3,0,0,:] = [0.427054, 0.715892]
        anchors[4,0,0,:] = [0.748154, 0.857092]
        outputs = outputs.reshape((ny,nx,5,6)).transpose([2,0,1,3])

        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx))

        grid = np.stack((yv, xv), 2).reshape((1, ny, nx, 2)).astype(np.float32)

        assert self.in_height / ny == self.in_width / nx
        BLOCK_SIZE = self.in_width / nx

        outputs[..., 0:2] = (sigmoid(outputs[..., 0:2]) + grid) * BLOCK_SIZE
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * anchors * BLOCK_SIZE
        outputs[..., 4:] = sigmoid(outputs[..., 4:])

        outputs[..., 0] = (outputs[..., 0] - outputs[..., 2] / 2).clip(0)  # x center to top-left x
        outputs[..., 1] = (outputs[..., 1] - outputs[..., 3] / 2).clip(0)  # y center to top-left y
        # print(outputs)
        outputs[..., :4] = np.round(outputs[..., :4]).astype(np.int32)
        outputs = outputs.reshape(-1, 6)
        boxes = outputs[..., :4]
        scores = outputs[..., 4]
        class_ids = outputs[..., 4:].argmax(-1)
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou).flatten()

        print(boxes[indices])

        boxes[..., [0, 2]] *= ratio[1]
        boxes[..., [1, 3]] *= ratio[0]

        # Draw detections that survived NMS
        [self.draw_detections(img, boxes[i], scores[i], class_ids[i]) for i in indices]

        return img

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
        y = self.model.get_tensor(self.out_index)
        if self.int8:
            y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        # Process detections and return result
        return self.postprocess(img, y, ratio)


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="../pretrained_models/coco_2017_person/tiny_yolo_v2_224_int8.tflite",
        help="Path to TFLite model.",
    )
    parser.add_argument("--img", type=str, default=str("../../../../public_dataset/COCO_person/000000000036.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.25, help="NMS IoU threshold")
    parser.add_argument("--metadata", type=str, default="../../../../public_dataset/COCO_person/coco80.yaml", help="Metadata yaml")
    args = parser.parse_args()
    detector = TinyYOLOv2TFLite(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(args.img)
    print("Show in result.jpg")
    cv2.imwrite("result.jpg", result)
    #cv2.imshow("Output", result)
    #cv2.waitKey(0)
