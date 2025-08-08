#!/bin/env python3
import sys
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

def evaluate_tflite_model(model_path, x_test, class_file):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], x_test)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    class_idx_and_prob = list(enumerate(output[0, :]))
    # print(class_idx_and_prob)
    sorted_class_idx_and_prob = sorted(class_idx_and_prob, key=lambda x: x[1], reverse=True)
    # print(sorted_class_idx_and_prob)
    with open(class_file, 'r') as f:
        classes = json.load(f)

    # top 3
    for i in range(3):
        idx, prob = sorted_class_idx_and_prob[i]
        # print(idx, prob)
        print(f"Label: {classes[str(idx)]}, Probability: {prob}")


if __name__ == "__main__":
    model_path = "../pretrained_models/ImageNet/resnet50_v2_224/resnet50_v2_224_int8.tflite"
    img = "../../../../public_dataset/imagenet/ILSVRC2012_img_val_samples/ILSVRC2012_val_00026100.JPEG"
    class_file = "../../../../public_dataset/imagenet/ILSVRC2012_img_val_samples/imagenet1000_clsidx_to_labels.json"
    target_h = 224
    target_w = 224
    x_test = load_and_resize_img(img, target_h, target_w)
    evaluate_tflite_model(model_path, x_test, class_file)

