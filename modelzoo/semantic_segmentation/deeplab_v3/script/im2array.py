#!/bin/env python3

import numpy as np
import cv2
import sys
from typing import Tuple, Union

def letterbox(img: np.ndarray, new_shape: Tuple = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float]]:
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

def preprocess(img: np.ndarray, new_shape: Tuple = (640, 640)) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Preprocesses the input image before performing inference.

    Args:
        img (np.ndarray): The input image to be preprocessed.

    Returns:
        Tuple[np.ndarray, Tuple[float, float]]: A tuple containing:
            - The preprocessed image (np.ndarray).
            - A tuple of two float values representing the padding applied (top/bottom, left/right).
    """
    img, pad = letterbox(img, new_shape)
    print(pad)
    img = img[..., ::-1][None]  # N,H,W,C for TFLite
    img = np.ascontiguousarray(img)
    img = img.astype(np.uint8)
    return img, pad

def load_and_resize_img(img, new_shape):
    img = cv2.imread(img)
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    print(f"img.shape is (%d %d), resize shape is {new_shape}" % (H, W))

    x_test, pad = preprocess(x_test, new_shape)

    print(f"x_test.shape is {x_test.shape}, Pad = {pad}")
 
    return x_test, pad

def im2array(data, c_file):
    test = list(data.flatten())
    with open (c_file, 'w') as f:
        header_line = "unsigned char imgArray[" + str(len(test)) + "]" + " {\n"
        f.write(header_line)

        for i in range(0, len(test), 12):
            chunk = test[i:i+12]
            line = ", ".join([str(hex(x)) for x in chunk])
            line += ","
            f.write(f"    {line}\n")

        f.write("};\n")

if __name__ == "__main__":
    img = "../../../public_dataset/COCO_person/000000000036.jpg"
    target_h = 256
    target_w = 256
    x_test, pad = load_and_resize_img(img, (target_h, target_w))
    c_file = "test_image_provider.cc"
    im2array(x_test, c_file)

