#!/bin/env python3

import sys
import librosa
import numpy as np
import tensorflow as tf

def quantize(input, scale, zero_point):
    return (input / scale) + zero_point

def dequantize(output, scale, zero_point):
    return (output.astype(np.float32) - zero_point) * scale

def normalize(values):
    return (values - np.mean(values)) / np.std(values)

def transform_audio_to_mfcc(audio_file, n_mfcc=13, n_fft=512, hop_length=160):
    audio_data, sample_rate = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # add derivatives and normalize
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfcc = np.concatenate((normalize(mfcc), normalize(mfcc_delta), normalize(mfcc_delta2)), axis=0)
    seq_length = mfcc.shape[1] // 2

    mfcc_out = mfcc.T.astype(np.float32)
    mfcc_out = np.expand_dims(mfcc_out, 0)

    return mfcc_out, seq_length

def file_to_vector_array(model_path, wavfile):
    input_window_length = 296

    data, data_length = transform_audio_to_mfcc(wavfile)

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_dtype = input_details["dtype"]
    output_dtype = output_details["dtype"]

    # Check if the input/output type is quantized,
    # set scale and zero-point accordingly
    if input_dtype != tf.float32:
        input_scale, input_zero_point = input_details["quantization"]
    else:
        input_scale, input_zero_point = 1, 0

    if output_dtype != tf.float32:
        output_scale, output_zero_point = output_details["quantization"]
    else:
        output_scale, output_zero_point = 1, 0

    data = quantize(data, input_scale, input_zero_point)
    # Round the data up if dtype is int8, uint8 or int16
    if input_dtype is not np.float32:
        data = np.round(data)
    print(data.shape)
    while data.shape[1] < input_window_length:
        data = np.append(data, data[:, -2:-1, :], axis=1)
    # Zero-pad any odd-length inputs
    if data.shape[1] % 2 == 1:
        # log('Input length is odd, zero-padding to even (first layer has stride 2)')
        data = np.concatenate([data, np.zeros((1, 1, data.shape[2]), dtype=input_dtype)], axis=1)
    data = np.array(data, dtype=input_details["dtype"])
    return data

def wav2array(data, c_file):
    test = list(data.flatten())
    with open (c_file, 'w') as f:
        header_line = "signed char wavArray[" + str(len(test)) + "]" + " = {\n"
        f.write(header_line)

        for i in range(0, len(test), 12):
            chunk = test[i:i+12]
            line = ", ".join([str(x) for x in chunk])
            line += ","
            f.write(f"    {line}\n")

        f.write("};\n")

if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    model_path = "../pretrained_models/wav2letter/wav2letter_int8.tflite"
    wav_file = "../../../../public_dataset/wav2letter/example_input.flac"

    x_test = file_to_vector_array(model_path, wav_file)
    c_file = "test_wav_provider.cc"
    wav2array(x_test, c_file)

