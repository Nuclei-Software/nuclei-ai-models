#!/bin/env python3

from PIL import Image, ImageDraw
import numpy as np
import cv2
import sys

def load_and_resize_img(img, target_h, target_w):
    img = cv2.imread(img)
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    print("img.shape is (%d %d), resize shape is (%d %d)" % (H, W, target_h, target_w))

    x_test = cv2.resize(x_test, (target_h, target_w)).astype(np.float32)
    x_test = x_test.astype(np.uint8)

    print(f"x_test.shape is {x_test.shape}")
 
    return x_test

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
    img = "../../../../public_dataset/cifar10/subset/51-8.jpg"
    target_h = 32
    target_w = 32
    x_test = load_and_resize_img(img, target_h, target_w)
    c_file = "test_image_provider.cc"
    im2array(x_test, c_file)

