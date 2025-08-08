import argparse
from typing import Tuple, Union

import cv2
import numpy as np
import tensorflow as tf
import yaml

# from ultralytics.utils import ASSETS


class lprnet:
    def __init__(self, imgdata:np.ndarray):
        self.model = '../pretrained_models/ocr_model_quant_int8.tflite'

        self.chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
            u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
            u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
            u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
            ]

        self.img=imgdata

    def fastdecode(self, y_pred):
        results = ""
        confidence = 0.0
        table_pred = y_pred.reshape(-1, len(self.chars)+1)
        # print(table_pred.shape)
        res = table_pred.argmax(axis=1)
        for i,one in enumerate(res):
            if one < len(self.chars) and (i==0 or (one!=res[i-1])):
                results+= self.chars[one]
                confidence+=table_pred[i][one]
        confidence/= len(results)
        return results, confidence

    def detect(self) -> np.ndarray:
        # set interpreter
        interpreter = tf.lite.Interpreter(model_path=self.model)
        interpreter.allocate_tensors()

        # get i/o tensor

        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]

        img = cv2.resize(self.img, (160, 40))

        img = img.transpose(1, 0, 2) #.astype(np.float32)
        img = img[np.newaxis, :, :, :]
        print(img.shape)

        # feed forward
        # get i/o tensor
        mean, std_dev = interpreter.get_output_details()[0]['quantization']

        interpreter.set_tensor(input_details['index'], img)

        interpreter.invoke()

        output = interpreter.get_tensor(output_details['index'])

        result = (output - mean) / std_dev

        # decode
        result = result[:,2:,:]
        return self.fastdecode(result)

class YOLOv5TFLite:
    """
    YOLOv5TFLite.

    A class for performing object detection using the YOLOv8 model with TensorFlow Lite.

    Attributes:
        model (str): Path to the TensorFlow Lite model file.
        conf (float): Confidence threshold for filtering detections.
        iou (float): Intersection over Union threshold for non-maximum suppression.
        metadata (Optional[str]): Path to the metadata file, if any.

    Methods:
        detect(img_path: str) -> np.ndarray:
            Performs inference and returns the output image with drawn detections.
    """

    def __init__(self, model: str, conf: float = 0.25, iou: float = 0.45, metadata: Union[str, None] = None):
        """
        Initializes an instance of the YOLOv5TFLite class.

        Args:
            model (str): Path to the TFLite model.
            conf (float, optional): Confidence threshold for filtering detections. Defaults to 0.25.
            iou (float, optional): IoU (Intersection over Union) threshold for non-maximum suppression. Defaults to 0.45.
            metadata (Union[str, None], optional): Path to the metadata file or None if not used. Defaults to None.
        """
        self.conf = conf
        self.iou = iou
        if metadata is None:
            self.classes = {i: i for i in range(1000)}
        else:
            with open(metadata) as f:
                self.classes = yaml.safe_load(f)["names"]
        np.random.seed(20)
        self.color_palette = np.random.uniform(128, 255, size=(len(self.classes), 3))

        self.model = tf.lite.Interpreter(model_path=model)
        self.model.allocate_tensors()

        input_details = self.model.get_input_details()[0]

        self.in_width, self.in_height = input_details["shape"][1:3]
        self.in_index = input_details["index"]
        self.in_scale, self.in_zero_point = input_details["quantization"]
        self.int8 = input_details["dtype"] == np.int8

        output_details = self.model.get_output_details()[0]
        self.out_index = output_details["index"]
        self.out_scale, self.out_zero_point = output_details["quantization"]

    def letterbox(self, img: np.ndarray, new_shape: Tuple = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float]]:
        """Resizes and reshapes images while maintaining aspect ratio by adding padding, suitable for YOLO models."""
        shape = img.shape[:2]  # current shape [height, width]

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        return img, (top / img.shape[0], left / img.shape[1])

    def draw_detections(self, img: np.ndarray, box: np.ndarray, score: np.float32, class_id: int) -> None:
        """
        Draws bounding boxes and labels on the input image based on the detected objects.

        Args:
            img (np.ndarray): The input image to draw detections on.
            box (np.ndarray): Detected bounding box in the format [x1, y1, width, height].
            score (np.float32): Corresponding detection score.
            class_id (int): Class ID for the detected object.

        Returns:
            None
        """
        x1, y1, w, h = box
        color = self.color_palette[class_id]

        cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

        label = f"{self.classes[class_id]}: {score:.2f}"

        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        label_x = x1
        label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10

        cv2.rectangle(
            img,
            (int(label_x), int(label_y - label_height)),
            (int(label_x + label_width), int(label_y + label_height)),
            color,
            cv2.FILLED,
        )

        print("Box (%d, %d, %d, %d), Score: %s" % (int(x1), int(y1), int(x1 + w), int(y1 + h), label))
        cv2.putText(img, label, (int(label_x), int(label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Preprocesses the input image before performing inference.

        Args:
            img (np.ndarray): The input image to be preprocessed.

        Returns:
            Tuple[np.ndarray, Tuple[float, float]]: A tuple containing:
                - The preprocessed image (np.ndarray).
                - A tuple of two float values representing the padding applied (top/bottom, left/right).
        """
        img, pad = self.letterbox(img, (self.in_width, self.in_height))
        img = img[...][None]  # N,H,W,C for TFLite

        img = np.ascontiguousarray(img)
        img = img.astype(np.uint8)
        return img, pad

    def postprocess(self, img: np.ndarray, outputs: np.ndarray, pad: Tuple[float, float]) -> np.ndarray:
        """
        Performs post-processing on the model's output to extract bounding boxes, scores, and class IDs.

        Args:
            img (numpy.ndarray): The input image.
            outputs (numpy.ndarray): The output of the model.
            pad (Tuple[float, float]): Padding used by letterbox.

        Returns:
            numpy.ndarray: The input image with detections drawn on it.
        """
        outputs[..., 0] -= pad[1]
        outputs[..., 1] -= pad[0]
        outputs[..., :4] *= max(img.shape)
        outputs[..., 0] -= outputs[..., 2] / 2
        outputs[..., 1] -= outputs[..., 3] / 2
    
        for out in outputs:
            scores = out[:, 4]

            keep = scores > self.conf
            # print(keep.shape)
            boxes = out[keep, :4]
            scores = scores[keep]
            # print(scores.shape)
            class_ids = out[keep, 4:].argmax(-1)

            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf, self.iou)
            if indices: 
                indices = indices.flatten()
            else:
                print("Couldn't detect what you focus!")
            
            for i in indices:
                x1, y1, w, h = boxes[i]
                print(f"position is (%d %d %d %d)"%(x1, y1, w, h))
                detector = lprnet(img[int(y1):int(y1+h), int(x1):int(x1+w), :])
                lprnet_res = detector.detect()
                print(lprnet_res)
                self.classes[class_ids[i]] = lprnet_res[0]
                self.draw_detections(img, boxes[i], scores[i], class_ids[i])

        return img

    def detect(self, img_path: str) -> np.ndarray:
        """
        Performs inference using a TFLite model and returns the output image with drawn detections.

        Args:
            img_path (str): The path to the input image file.

        Returns:
            np.ndarray: The output image with drawn detections.
        """
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x, pad = self.preprocess(img)
        if self.int8:
            x = (x / self.in_scale + self.in_zero_point).astype(np.int8)

        self.model.set_tensor(self.in_index, x)

        self.model.invoke()

        y = self.model.get_tensor(self.out_index)

        #if self.int8:
        y = (y.astype(np.float32) - self.out_zero_point) * self.out_scale
        # print(y)

        return self.postprocess(img, y, pad)


if __name__ == "__main__":
    np.set_printoptions(threshold=np.inf, linewidth=200)
    model_path = "../pretrained_models/yolov5_best-int8.tflite"
    img = "../../../../public_dataset/LPRNET/image_rec/1.jpg"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=model_path,
        help="Path to TFLite model.",
    )
    parser.add_argument("--img", type=str, default=str(img), help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--metadata", type=str, default=None, help="Metadata yaml")
    args = parser.parse_args()

    detector = YOLOv5TFLite(args.model, args.conf, args.iou, args.metadata)
    result = detector.detect(args.img)
    print("Show in result.jpg")
    cv2.imwrite("result.jpg", result)
    #cv2.imshow("Output", result)
    #cv2.waitKey(0)
