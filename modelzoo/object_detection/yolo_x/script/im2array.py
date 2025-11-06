#!/bin/env python3

import numpy as np
import cv2
import sys
from typing import Tuple, Union

def load_and_resize_img(img_path, new_shape):
    img = cv2.imread(img_path)
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    print(f"imgage shape is (%d %d), resize shape is {new_shape}" % (H, W))

    x_test = cv2.resize(x_test, (target_h, target_w)).astype(np.float32)
    x_test = x_test.astype(np.int8)

    ratio = (H / target_h, W / target_w)

    print(f"x_test.shape is {x_test.shape}, Ratio = {ratio}")
 
    return x_test, ratio

def im2array(data, c_file):
    test = list(data.flatten())
    with open (c_file, 'w') as f:
        header_line = "signed char imgArray[" + str(len(test)) + "]" + " {\n"
        f.write(header_line)

        for i in range(0, len(test), 12):
            chunk = test[i:i+12]
            line = ", ".join([str(hex(x)) for x in chunk])
            line += ","
            f.write(f"    {line}\n")

        f.write("};\n")

if __name__ == "__main__":
    img_path = "../../../../public_dataset/COCO_person/000000000036.jpg"
    target_h = 256
    target_w = 256
    x_test, ratio = load_and_resize_img(img_path, (target_h, target_w))
    c_file = "test_image_provider.cc"
    im2array(x_test, c_file)
