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
import matplotlib.pyplot as plt
from matplotlib import gridspec, pyplot as plt

# from ultralytics.utils import ASSETS

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter

def preprocess_image(img: np.ndarray = None, height: int = None, width: int = None, aspect_ratio: str = None,
                     interpolation: str = None, scale: float = None, offset: int = None,
                     perform_scaling: bool = True) -> tf.Tensor:

    """
    Predicts a class for all the images that are inside a given directory.
    The model used for the predictions can be either a .h5 or .tflite file.

    Args:
        img (np.ndarray): image to be prepared
        height (int): height in pixels
        width (int): width in pixels
        aspect_ratio (str): "fit' or "crop"
        interpolation (str): resizing interpolation method
        scale (float): rescaling pixels value
        offset (int): offset value on pixels
        perform_scaling (bool): whether to rescale or not the image

    Returns:
        img_processed (tf.Tensor): the prepared image

    """

    if aspect_ratio == "fit":
        img = tf.image.resize(img, [height, width], method=interpolation, preserve_aspect_ratio=False)
    else:
        img = tf.image.resize_with_crop_or_pad(img, height, width)

    # Rescale the image
    if perform_scaling:
        img_processed = scale * tf.cast(img, tf.float32) + offset
    else:
        img_processed = img

    return img_processed

def preprocess_input(image: np.ndarray, input_details: dict) -> tf.Tensor:
    """
    Preprocesses an input image according to input details.

    Args:
        image: Input image as a NumPy array.
        input_details: Dictionary containing input details, including quantization and dtype.

    Returns:
        Preprocessed image as a TensorFlow tensor.

    """

    # Get the dimensions
    if input_details is not None:
        if input_details['dtype'] in [np.uint8, np.int8]:
            image_processed = (image / input_details['quantization'][0]) + input_details['quantization'][1]
            image_processed = np.clip(np.round(image_processed), np.iinfo(input_details['dtype']).min,
                                      np.iinfo(input_details['dtype']).max)
        else:
            image_processed = image
        image_processed = tf.cast(image_processed, dtype=input_details['dtype'])
    else:
        image_processed = image

    image_processed = tf.expand_dims(image_processed, 0)

    return image_processed

COLOR_MAP = {
    (0, 0, 0): 0,          # background
    (128, 0, 0): 1,        # aeroplane
    (0, 128, 0): 2,        # bicycle
    (128, 128, 0): 3,      # bird
    (0, 0, 128): 4,        # boat
    (128, 0, 128): 5,      # bottle
    (0, 128, 128): 6,      # bus
    (128, 128, 128): 7,    # car
    (64, 0, 0): 8,         # cat
    (192, 0, 0): 9,        # chair
    (64, 128, 0): 10,      # cow
    (192, 128, 0): 11,     # dining table
    (64, 0, 128): 12,      # dog
    (192, 0, 128): 13,     # horse
    (64, 128, 128): 14,    # motorbike
    (192, 128, 128): 15,   # person
    (0, 64, 0): 16,        # potted plant
    (128, 64, 0): 17,      # sheep
    (0, 192, 0): 18,       # sofa
    (128, 192, 0): 19,     # train
    (0, 64, 128): 20       # tv/monitor
}


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark."""
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
    return colormap


def label_to_color_image(label: np.ndarray = None) -> np.ndarray:

    """
    Adds color defined by the dataset colormap to the label.
    Args:
        label (np.ndarray): vector of labels we want to map into colors
    Returns:
        np.ndarray: colors associated to labels
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
    colormap = create_pascal_label_colormap()
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
    return colormap[label]

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

    def vis_segmentation(self, image_path: str = None, seg_map: np.ndarray = None,
                        input_size: list = None):

        """
            Predicts a class for all the images that are inside a given directory.
            The model used for the predictions can be either a .h5 or .tflite file.

            Args:
                image_path (str): complete path to input image
                seg_map (np.ndarray): segmentation class output: each pixel is associated to a class
                cfg (dict): A dictionary containing the configuration file parameters.
                input_size (list): [height, width] in pixels

            Returns:

        """

        # Some instrumental parameters
        # Load class names if class_file is provided
        class_names = self.classes
        nb_channels = 3
        interpolation = 'bilinear'
        aspect_ratio = 'fit'

        height = input_size[0]
        width = input_size[1]

        # directory for saving prediction outputs
        prediction_result_dir = f'./'

        # Load the original image
        original_image = tf.io.read_file(image_path)
        original_image = tf.image.decode_image(original_image, channels=nb_channels)   

        original_image = preprocess_image(original_image, height=height, width=width, aspect_ratio=aspect_ratio, 
                                        interpolation=interpolation, scale=None, offset=None, perform_scaling=False)

        plt.ioff()

        # Visualize the segmentation
        full_label_map = np.arange(len(class_names)).reshape(len(class_names), 1)
        full_color_map = label_to_color_image(full_label_map)

        plt.figure(figsize=(15, 5))
        grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

        # back to integer numbers for plotting
        original_image = original_image.numpy()
        original_image = original_image.astype(np.uint8)
        # Plot input image
        plt.subplot(grid_spec[0])
        plt.imshow(original_image)
        plt.axis('off')
        plt.title('Input image')

        # plot Segmentation map
        plt.subplot(grid_spec[1])
        seg_image = label_to_color_image(seg_map).astype(np.uint8)
        plt.imshow(seg_image)
        plt.axis('off')
        plt.title('Segmentation map')

        # plot input and segmentation overlay
        plt.subplot(grid_spec[2])
        extent = [0, input_size[0], input_size[1], 0]
        plt.imshow(original_image, extent=extent)
        plt.imshow(seg_image, alpha=0.7, extent=extent)
        plt.axis('off')
        plt.title('Segmentation overlay')

        unique_labels = np.unique(seg_map)
        ax = plt.subplot(grid_spec[3])
        plt.imshow(full_color_map[unique_labels].astype(np.uint8), interpolation='nearest')
        ax.yaxis.tick_right()
        plt.yticks(range(len(unique_labels)), [class_names[i] for i in unique_labels])
        plt.xticks([], [])
        ax.tick_params(width=0.0)
        plt.grid('off')

        # Save figure in the predictions directory
        print("Show in result.jpg")
        plt.savefig('result.png', bbox_inches='tight')


        # plt.waitforbuttonpress()

        # plt.close()


    def postprocess_output_values(self, output: np.ndarray, output_details: dict) -> np.ndarray:
        """
        Postprocesses the model output to obtain the predicted label.

        Args:
            output (np.ndarray): The output tensor from the model.
            output_details: Dictionary containing output details, including quantization and dtype.

        Returns:
            np.ndarray: The predicted label.
        """
        if output_details is not None:
            if output_details['dtype'] in [np.uint8, np.int8]:
                # Convert the output data to float32 data type and perform the inverse quantization operation
                predicted_label = (output - output_details['quantization'][1]) * output_details['quantization'][0]
            else:
                predicted_label = output
        else:
            predicted_label = output

        return predicted_label

    def generate_output_image(self, image_path: str = None, output: np.ndarray = None,
                            input_size: list = None):
        """
        Post-processing to convert raw output to segmentation output and then display input image with segmentation overlay

        Args:
            image_path (str): path to the network input image
            output (np.ndarray): corresponding network output
            cfg (dict): A dictionary containing the entire configuration file.
            input_size (list): [height, width] of the input image

        Returns:
            None
        """

        # at this stage input image and network output are always channel last (so axis=3)
        seg_map = tf.argmax(tf.image.resize(output, size=input_size), axis=3)
        seg_map = tf.squeeze(seg_map).numpy().astype(np.int8)

        self.vis_segmentation(image_path=image_path, seg_map=seg_map, input_size=input_size)
    def detect(self, img_path: str) -> np.ndarray:
        """
        Perform object detection on an input image.

        Args:
            img_path (str): Path to the input image file.

        Returns:
            (np.ndarray): The output image with drawn detections.
        """
        input_details = self.model.get_input_details()[0]


        height, width, _ = input_details['shape_signature'][1:]

        # Load and preprocess image
        aspect_ratio = 'fit'
        interpolation = 'bilinear'
        scale = 1/127.5
        offset = -1
        channels = 3
        # Load the image
        try:
            data = tf.io.read_file(img_path)
            img = tf.image.decode_image(data, channels=channels)
        except:
            raise ValueError(f"\nUnable to load image file {img_path}\n"
                             "Supported image file formats are BMP, GIF, JPEG and PNG.")

        img = preprocess_image(img, height=height, width=width, aspect_ratio=aspect_ratio, interpolation=interpolation,
                               scale=scale, offset=offset, perform_scaling=True)
        x = preprocess_input(img, input_details=input_details)

        # Apply quantization if model is int8
        # if self.int8:
        #     x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        # Set input tensor and run inference
        self.model.set_tensor(self.in_index, x)
        self.model.invoke()

        # # Get output and dequantize if necessary
        # y = self.model.get_tensor(self.out_index)
        # if self.int8:
        #     y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale

        # Process detections and return result

        # generation of output images
        # Get output details
        output_details = self.model.get_output_details()[0]

        output_index_quant = output_details["index"]
        raw_prediction = self.model.get_tensor(output_index_quant)
        output = self.postprocess_output_values(output=raw_prediction, output_details=output_details)
    
        self.generate_output_image(image_path=img_path, output=output, input_size=[height, width])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="../pretrained_models/coco_2017_pascal_voc_2012/deeplab_v3_mobilenetv2_05_16_512/deeplab_v3_mobilenetv2_05_16_512_asppv1_int8.tflite",
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
