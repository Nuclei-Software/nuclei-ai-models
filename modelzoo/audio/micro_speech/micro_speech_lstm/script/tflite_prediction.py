#!/bin/env python3
import sys
import numpy as np
import tensorflow as tf
import dataset

LABELS = ["right", "go", "no", "left", "stop", "up", "down", "yes"]
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
    # print(spectrogram)
    interpreter.set_tensor(input_details[0]["index"], spectrogram)
    interpreter.invoke()
    model_predictions = interpreter.get_tensor(output_details[0]["index"])
    # print(model_predictions)
    tflite_label = np.argmax(model_predictions)
    print("Recognized word: ", LABELS[tflite_label])


if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    model_path = "../pretrained_models/microspeech_lstm/microspeech_lstm_int8.tflite"
    wav_file = "../../../../../public_dataset/micro_speech/down/0a9f9af7_nohash_0.wav"
    evaluate_tflite_model(model_path, wav_file)
