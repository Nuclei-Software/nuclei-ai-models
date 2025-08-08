#!/bin/env python3

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 加载MNIST数据集
def load_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 将图像转换为float32类型，并缩放到[0, 255]范围
    x_test = x_test.astype(np.float32)
    # 对于uint8模型，我们通常需要将数据归一化到[0, 255]
    x_test = ((x_test / 255.0) * 255).astype(np.uint8)
    # TensorFlow Lite期望输入形状为[N, 28, 28, 1]
    x_test = np.expand_dims(x_test, axis=-1)
    return x_test, y_test

# 测试模型的准确度
def evaluate_tflite_model(model_path, x_test, y_test):
    # 加载TFLite模型
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # 获取输入输出张量的信息
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    correct_predictions = 0
    for i in range(len(x_test)):
        # 预处理输入
        test_image = np.expand_dims(x_test[i], axis=0)
     
        # 设置输入
        interpreter.set_tensor(input_details['index'], test_image)

        # 运行推理
        interpreter.invoke()

        # 获取输出结果
        output = interpreter.get_tensor(output_details['index'])
        prediction = np.argmax(output)

        if prediction == y_test[i]:
            correct_predictions += 1

    accuracy = correct_predictions / len(y_test)
    print(f"Model accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    model_path = "../pretrained_models/mnist_int8.tflite"
    x_test, y_test = load_data()
    evaluate_tflite_model(model_path, x_test, y_test)
