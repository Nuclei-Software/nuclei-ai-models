#!/bin/env python3

import cv2
import numpy as np
import tensorflow as tf
import json

def load_and_resize_img(img, target_h, target_w):
    img = cv2.imread(img)
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    print("img.shape is (%d %d), resize shape is (%d %d)" % (H, W, target_h, target_w))

    x_test = cv2.resize(x_test, (target_h, target_w)).astype(np.float32)
    x_test = x_test.astype(np.uint8)
    x_test = np.expand_dims(x_test, axis=0)

    print(f"x_test.shape is {x_test.shape}")
 
    return x_test

class_names = [ "daisy", "dandelion", "roses", "sunflowers", "tulips" ]
def evaluate_tflite_model(model_path, x_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], x_test)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    output = output[0, :]

    idx = np.argmax(output)

    print(f"The result is: {class_names[idx]} Probability:{output[idx]}")
    


if __name__ == "__main__":
    model_path = "../pretrained_models/flowers/fdmobilenet_0.25_128_tfs_int8.tflite"
    img = "../../../../public_dataset/flower_photos/roses/568715474_bdb64ccc32.jpg"
    target_h = 128
    target_w = 128
    x_test = load_and_resize_img(img, target_h, target_w)
    evaluate_tflite_model(model_path, x_test)

