# MicroSpeech model with LSTM

## Introduction

MicroSpeech model is a basic speech recognition network, that recognize two words: 'yes' and 'no'. All other words are
classified as 'unknown'. This model was designed as The TensorFlow tutorial on speech recognition networks. The tutorial
is available
on [Google Colab](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/third_party/xtensa/examples/micro_speech_lstm/train/micro_speech_with_lstm_op.ipynb)

The model was trained on the Speech Commands dataset [1]. It is an audio dataset of spoken words, for training models
that detect when a single word is spoken. It’s released under
a [Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).

## Model Information

 Information      | Value                                                        
 ---------------- | ------------------------------------------------------------ 
 Input shape      | Wav sound file, with shape (1, 49, 257)                      
 Output shape     | Vector of probabilities, shape (1, 8). Labels are["right", "go", "no", "left", "stop", "up", "down", "yes"] 
 FLOPS            | 0.062 MOPS                                                   
 Source framework | Tensorflow/Keras                                             
 Target platform  | MPU                                                          

## Version and changelog

Initial release of quantized int8 model.

### Labels

- 'right'
- 'go'
- 'no'
- 'left'
- 'stop'
- 'up'
- 'down'
- 'yes'

## Origin

Model
implementation: [Google colab](https://colab.research.google.com/github/tensorflow/tflite-micro/blob/main/third_party/xtensa/examples/micro_speech_lstm/train/micro_speech_with_lstm_op.ipynb)

[1] Warden, Pete. "Speech commands: A dataset for limited-vocabulary speech recognition." arXiv preprint arXiv:
1804.03209 (2018).
