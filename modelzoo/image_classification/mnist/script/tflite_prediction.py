#!/bin/env python3

import cv2
import numpy as np
import tensorflow as tf

def load_and_resize_img(img, target_h, target_w):
    img = cv2.imread(img)
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W, _ = img.shape
    print("img.shape is (%d %d)" % (H, W))

    x_test = cv2.resize(x_test, (target_h, target_w)).astype(np.float32)
    x_test = x_test.astype(np.uint8)

    x_test = np.expand_dims(x_test, axis=-1)
    x_test = np.expand_dims(x_test, axis=0)

    print(f"x_test.shape is {x_test.shape}")
 
    return x_test

class_names = [ '0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
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
    # np.set_printoptions(threshold=sys.maxsize)
    model_path = "../pretrained_models/mnist_int8.tflite"
    img = "../../../../public_dataset/quant_img_mnist/0.bmp"
    target_h = 28
    target_w = 28
    x_test = load_and_resize_img(img, target_h, target_w)
    evaluate_tflite_model(model_path, x_test)

