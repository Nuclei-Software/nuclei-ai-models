#!/bin/env python3

import cv2
import numpy as np
import tensorflow as tf

def load_and_resize_img(img, target_h, target_w):
    img = cv2.imread(img)
    x_test = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W, _ = img.shape
    print("img.shape is (%d %d), resize shape is (%d %d)" % (H, W, target_h, target_w))

    x_test = cv2.resize(x_test, (target_h, target_w)).astype(np.float32)
    x_test = x_test.astype(np.uint8)
    x_test = x_test.transpose(1, 0, 2)
    x_test = np.expand_dims(x_test, axis=0)

    print(f"x_test.shape is {x_test.shape}")
 
    return x_test


chars = [u"京", u"沪", u"津", u"渝", u"冀", u"晋", u"蒙", u"辽", u"吉", u"黑", u"苏", u"浙", u"皖", u"闽", u"赣", u"鲁", u"豫", u"鄂", u"湘", u"粤", u"桂",
         u"琼", u"川", u"贵", u"云", u"藏", u"陕", u"甘", u"青", u"宁", u"新", u"0", u"1", u"2", u"3", u"4", u"5", u"6", u"7", u"8", u"9", u"A",
         u"B", u"C", u"D", u"E", u"F", u"G", u"H", u"J", u"K", u"L", u"M", u"N", u"P", u"Q", u"R", u"S", u"T", u"U", u"V", u"W", u"X",
         u"Y", u"Z",u"港",u"学",u"使",u"警",u"澳",u"挂",u"军",u"北",u"南",u"广",u"沈",u"兰",u"成",u"济",u"海",u"民",u"航",u"空"
         ]

def fastdecode(y_pred):
    results = ""
    confidence = 0.0
    table_pred = y_pred.reshape(-1, len(chars)+1)
    print(table_pred.shape)
    res = table_pred.argmax(axis=1)
    for i,one in enumerate(res):
        if one < len(chars) and (i==0 or (one!=res[i-1])):
            results+= chars[one]
            confidence+=table_pred[i][one]
    confidence/= len(results)
    return results, confidence

def evaluate_tflite_model(model_path, x_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # get i/o tensor
    mean, std_dev = interpreter.get_output_details()[0]['quantization']

    interpreter.set_tensor(input_details['index'], x_test)

    interpreter.invoke()

    output = interpreter.get_tensor(output_details['index'])

    result = (output - mean) / std_dev

    # decode
    result = result[:,2:,:]
    print(fastdecode(result))

if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    model_path = "../pretrained_models/ocr_model_quant_int8.tflite"
    img = "../../../../public_dataset/LPRNET/ocr_images/0.jpg"
    target_h = 160
    target_w = 40
    x_test = load_and_resize_img(img, target_h, target_w)
    evaluate_tflite_model(model_path, x_test)

