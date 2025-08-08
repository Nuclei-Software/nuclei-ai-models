#!/bin/env python3

import numpy as np
import dataset
import tensorflow as tf
import sys

def file_to_vector_array(model_path, x_test):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zero_point = input_details[0]["quantization"]
    out_scale, out_zero_point = output_details[0]["quantization"]
    print("==== conf ===")
    print(f"in_scale:{in_scale}, in_zero_point:{in_zero_point}")
    print(f"out_scale:{in_scale}, out_zero_point:{in_zero_point}")
    print("==== conf end ===")
    in_dtype = input_details[0]["dtype"]
    is_in_quantized = in_dtype != np.float32

    audio_binary = tf.io.read_file(x_test)
    waveform = dataset.decode_audio(audio_binary)
    spectrogram = dataset.get_spectrogram(waveform)
    spectrogram = np.array(spectrogram)

    if is_in_quantized:
        spectrogram = dataset.quantize(
            spectrogram, in_scale, in_zero_point, in_dtype
        )

    spectrogram = np.array(spectrogram, dtype=input_details[0]["dtype"]).reshape(
        1, 49, 257
    )
    return spectrogram

def wav2array(data, c_file):
    test = list(data.flatten())
    with open (c_file, 'w') as f:
        header_line = "signed char wavArray[" + str(len(test)) + "]" + " {\n"
        f.write(header_line)

        for i in range(0, len(test), 12):
            chunk = test[i:i+12]
            line = ", ".join([str(x) for x in chunk])
            line += ","
            f.write(f"    {line}\n")

        f.write("};\n")

if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    model_path = "../pretrained_models/microspeech_lstm/microspeech_lstm_int8.tflite"
    wav_file = "../../../../../public_dataset/micro_speech/down/0a9f9af7_nohash_0.wav"

    x_test = file_to_vector_array(model_path, wav_file)
    c_file = "test_wav_provider.cc"
    wav2array(x_test, c_file)

