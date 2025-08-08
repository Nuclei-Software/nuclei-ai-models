#!/bin/env python3

import cv2
import numpy as np
import tensorflow as tf

def draw_keypoints(image, keypoints):
    print(keypoints.shape)
    for keypoint in keypoints:
        print(keypoint)
        y, x, _ = keypoint
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
        
    return image


def evaluate_tflite_model(model_path, img_path):

    target_h = 192
    target_w = 192

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    H, W, _ = img_rgb.shape
    print("img.shape is (%d %d)" % (H, W))

    x_test = cv2.resize(img_rgb, (target_h, target_w)).astype(np.float32)
    x_test = x_test.astype(np.uint8)
    x_test = np.expand_dims(x_test, axis=0)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    #print(x_test)

    interpreter.set_tensor(input_details['index'], x_test)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details['index'])

    data = output[0, 0]  # shape: (17, 3)

    data[:, 0] *= H
    data[:, 1] *= W

    output_image = draw_keypoints(img, data)
    return output_image

if __name__ == "__main__":
    model_path = "../pretrained_models/COCO_Person/movenet_lightning_192/movenet_singlepose_lightning_192_int8.tflite"
    img = "../../../../public_dataset/movenet/0000001.jpg"
    result = evaluate_tflite_model(model_path, img)
    print("Show in result.jpg")
    cv2.imwrite("result.jpg", result)