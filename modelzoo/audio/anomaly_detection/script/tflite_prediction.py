#!/bin/env python3

import sys

import librosa
import numpy as np
import tensorflow as tf

def file_load(wav_name):
    """
    load .wav file.
    wav_name : str
        target .wav file
    sampling_rate : int
        audio file sampling_rate
    return : np.array( float )
    """
    try:
        return librosa.load(wav_name, sr=None, mono=False)
    except:
        print("file_broken or not exists!! : {}".format(wav_name))


def quantize(input, scale, zero_point):
    return (input / scale) + zero_point

def dequantize(output, scale, zero_point):
    return (output - zero_point) * scale

def file_to_vector_array(
    file_name,
    n_mels=128,
    frames=5,
    n_fft=1024,
    hop_length=512,
    power=2.0,
    method="librosa",
):
    """
    convert file_name to a vector array.

    file_name : str
        target .wav file

    return : np.array( np.array( float ) )
        vector array
        * dataset.shape = (dataset_size, feature_vector_length)
    """
    # 01 calculate the number of dimensions
    dims = n_mels * frames

    # 02 generate melspectrogram
    y, sr = file_load(file_name)
    if method == "librosa":
        # 02a generate melspectrogram using librosa
        mel_spectrogram = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=power
        )

        # 03 convert melspectrogram to log mel energy
        log_mel_spectrogram = (
            20.0 / power * np.log10(mel_spectrogram + sys.float_info.epsilon)
        )

    else:
        print("spectrogram method not supported: {}".format(method))
        return np.empty((0, dims))

    # 3b take central part only
    log_mel_spectrogram = log_mel_spectrogram[:, 50:250]

    # 04 calculate total vector size
    vector_array_size = len(log_mel_spectrogram[0, :]) - frames + 1

    # 05 skip too short clips
    if vector_array_size < 1:
        return np.empty((0, dims))

    # 06 generate feature vectors by concatenating multiframes
    vector_array = np.zeros((vector_array_size, dims))
    for t in range(frames):
        vector_array[:, n_mels * t : n_mels * (t + 1)] = log_mel_spectrogram[
            :, t : t + vector_array_size
        ].T

    return vector_array

def evaluate_tflite_model(model_path, x_test):

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero_point = input_details[0]["quantization"]
    out_scale, out_zero_point = output_details[0]["quantization"]
    print("==== quantized conf ===")
    print(f"in_scale:{in_scale}, in_zero_point:{in_zero_point}")
    print(f"out_scale:{out_scale}, out_zero_point:{out_zero_point}")
    print("==== quantized conf end ===")

    data_fp = file_to_vector_array(x_test)
    data = quantize(data_fp, in_scale, in_zero_point)

    input_data = data.astype(dtype=np.int8)

    output_data = np.empty_like(input_data)

    for i in range(input_data.shape[0]):
        interpreter.set_tensor(input_details[0]["index"], input_data[i : i + 1, :])
        interpreter.invoke()

        output_data[i : i + 1, :] = interpreter.get_tensor(output_details[0]["index"])

    out = output_data.astype(np.float32)
    out = dequantize(out, out_scale, out_zero_point)
    errors = np.mean(np.square(data_fp - out), axis=1)
    y_pred = np.mean(errors)

    print("Prediction ", y_pred)


if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    model_path = "../pretrained_models/deep-autoencoder/ad01_int8.tflite"
    wav_file = "../../../../public_dataset/anomaly_detection/DCASE_2020_Challenge_Task2/ToyCar/id_05_00000010.wav"
    evaluate_tflite_model(model_path, wav_file)