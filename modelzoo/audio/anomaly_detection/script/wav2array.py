#!/bin/env python3
import sys

import librosa
import numpy as np

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

def wav2array(data, c_file):
    test = list(data.flatten())
    with open (c_file, 'w') as f:
        header_line = "float wavArray[" + str(len(test)) + "]" + " {\n"
        f.write(header_line)

        for i in range(0, len(test), 12):
            chunk = test[i:i+12]
            line = ", ".join([str(x) for x in chunk])
            line += ","
            f.write(f"    {line}\n")

        f.write("};\n")

if __name__ == "__main__":
    # np.set_printoptions(threshold=sys.maxsize)
    wav_file = "../../../../public_dataset/anomaly_detection/DCASE_2020_Challenge_Task2/ToyCar/id_05_00000010.wav"
    x_test = file_to_vector_array(wav_file)
    c_file = "test_wav_provider.cc"
    wav2array(x_test, c_file)

