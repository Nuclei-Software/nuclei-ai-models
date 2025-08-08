# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# See https://github.com/triple-Mu/yolov8/blob/main/examples/YOLOv8-TFLite-Python/main.py

import argparse
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import yaml

skeleton_connections_dict = {478:[],
                             21:[[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]],
                             17:[[0,1],[0,2],[1,3],[2,4],[3,5],[4,6],[5,7],[6,8],[7,9],[8,10],[5,6],[5,11],[6,12],[11,12],[11,13],[12,14],[13,15],[14,16]],
                             13:[[0,1],[0,2],[1,2],[1,3],[2,4],[3,5],[4,6],[1,7],[2,8],[7,9],[8,10],[9,11],[10,12],[7,8]]}

# from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

def padded_nms(tensor:tf.Tensor,max_output_size:int,iou_threshold:float,score_threshold:float):
    '''
    Function used to apply NMS on each image of the batch independently

    Args
        tensor          (tf.Tensor): shape (5+keypoints*3, num_boxes) FLOAT32 outputs of the YOLO multi pose estimation models
        max_output_size (tf.Tensor): shape (1,) INT32 max number of detections per image
        iou_threshold   (tf.Tensor): shape (1,) FLOAT32 threshold for NMS iou
        score_threshold (tf.Tensor): shape (1,) FLOAT32 threshold to filter detections under a certain score
    
    Returns:
        detection       (tf.Tensor): shape (max_output_size, 5+keypoints*3) FLOAT32 bounding boxes + (x,y,conf) values for all keypoints of all detected persons
    '''

    boxes     = tensor[:4] # shape : (4,num_boxes)
    scores    = tensor[4]  # shape : (num_boxes)
    keypoints = tensor[5:] # shape : (nb_kpts*3,num_boxes)
    # the scores must be of the form [y1, x1, y2, x2] in order for the NMS to work
    # they are initially of the form [x,y,w,h]

    x1 = boxes[0] - boxes[2]/2 # shape : (num_boxes)
    y1 = boxes[1] - boxes[3]/2 # shape : (num_boxes)
    x2 = boxes[0] + boxes[2]/2 # shape : (num_boxes)
    y2 = boxes[1] + boxes[3]/2 # shape : (num_boxes)

    yxboxes = tf.stack([y1, x1, y2, x2])  # shape : (4,num_boxes)
    yxboxes = tf.transpose(yxboxes,[1,0]) # shape : (num_boxes,4)

    selected_indices = tf.image.non_max_suppression(yxboxes,scores,
                                                    max_output_size = max_output_size,
                                                    iou_threshold   = iou_threshold,
                                                    score_threshold = score_threshold)
    indices = selected_indices.shape[0]

    if indices:
        detection = tf.gather(tensor,selected_indices,axis=1) # shape : (5+nb_kpts*3,selected_indices)
        detection = tf.transpose(detection,[1,0])             # shape : (selected_indices,5+nb_kpts*3)
    else:
        detection = tf.zeros((indices,tensor.shape[0]))

    zero_padding = tf.zeros((max_output_size-indices,tensor.shape[0])) # shape : (max_output_size-selected_indices,5+nb_kpts*3)

    detection = tf.concat([detection,zero_padding],0) # shape : (max_output_size,5+nb_kpts*3)
    return detection # shape : (max_output_size, 5+nb_kpts*3)

def yolo_mpe_postprocess(tensor:tf.Tensor,max_output_size:int=20,iou_threshold:float=0.7,score_threshold:float=0.25):
    '''
    Post-process for the multi pose estimation use-case

    Args
        tensor          (tf.Tensor): shape (batch, 5+keypoints*3, num_boxes) FLOAT32 outputs of the YOLO multi pose estimation models
        max_output_size (tf.Tensor): shape (1,) INT32 max number of detections per image
        iou_threshold   (tf.Tensor): shape (1,) FLOAT32 threshold for NMS iou
        score_threshold (tf.Tensor): shape (1,) FLOAT32 threshold to filter detections under a certain score
    
    Returns:
        detections      (tf.Tensor): shape (batch, max_output_size, 5+keypoints*3) FLOAT32 values for all keypoints of all detected persons
    '''


    tensor = tf.constant(tensor,tf.float32)

    args = {'max_output_size':max_output_size or 20,
            'iou_threshold':iou_threshold or 0.7,
            'score_threshold':score_threshold or 0.25}

    detections = tf.map_fn(lambda x : padded_nms(x,**args),tensor)

    return detections # shape : (batch, max_output_size, 5+keypoints*3)

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

    def postprocess(self, image: np.ndarray, outputs: np.ndarray, ratio: Tuple[float, float]) -> np.ndarray:
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

        poses = yolo_mpe_postprocess(outputs)[0]
        kpts_nbr = 17
        try:
            skeleton_connections = skeleton_connections_dict[kpts_nbr]
        except:
            print('Skeleton for this number of keypoints is not supported -> use 21, 17 or 13')

        threshSkeleton = 0.15
        width = self.in_width * ratio[1]
        height = self.in_height * ratio[0]

        bbox_thick = int(0.6 * (height + width) / 600)

        for ids,p in enumerate(poses):
            x,y,w,h,conf = p[:5]
            xx, yy, pp = p[5::3],p[5+1::3],p[5+2::3]
            x1 = int((x - w/2)*width)
            x2 = int((x + w/2)*width)
            y1 = int((y - h/2)*height)
            y2 = int((y + h/2)*height)

            if not tf.reduce_all(tf.constant(pp)==0):
                for i in range(0,len(xx)):
                    if float(pp[i])>threshSkeleton:
                        cv2.circle(image,(int(xx[i]*width),int(yy[i]*height)),radius=5,color=(0, 0, 255), thickness=-1)
                    else:
                        cv2.circle(image,(int(xx[i]*width),int(yy[i]*height)),radius=5,color=(255, 0, 0), thickness=-1)
                for k,l in skeleton_connections:
                    if float(pp[k])>threshSkeleton and float(pp[l])>threshSkeleton: 
                        cv2.line(image,(int(xx[k]*width),int(yy[k]*height)),(int(xx[l]*width),int(yy[l]*height)),(0, 255, 0))

            btext = '{}-{:.2f}'.format('person',conf)
            cv2.rectangle(image,(x1,y1), (x2, y2),(255, 0, 255),1)
            cv2.putText(image, btext, (x1,y1-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), bbox_thick//2, lineType=cv2.LINE_AA)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # writing prediction result to the output dir
        print("Show in result.jpg")
        cv2.imwrite("result.jpg", image)


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="../pretrained_models/yolov8n_256_quant_pc_uf_pose_coco-st.tflite",
        help="Path to TFLite model.",
    )
    parser.add_argument("--img", type=str, default=str("../../../../public_dataset/movenet/0000001.jpg"), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--metadata", type=str, default="../../../../public_dataset/COCO_person/coco80.yaml", help="Metadata yaml")
    args = parser.parse_args()
    detector = YOLOv8TFLite(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(args.img)
    # print("Show in result.jpg")
    # cv2.imwrite("result.jpg", result)
    #cv2.imshow("Output", result)
    #cv2.waitKey(0)
