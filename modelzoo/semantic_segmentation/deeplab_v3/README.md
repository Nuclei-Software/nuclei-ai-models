# DeepLab v3

## **Use case** : `Semantic Segmentation`

# Model description

DeepLabv3 was specified in "Rethinking Atrous Convolution for Semantic Image Segmentation" paper by Google.
It is composed by a backbone (encoder) that can be a Mobilenet V2 (width parameter alpha) or a ResNet-50 or 101 for example followed by an ASPP (Atrous Spatial Pyramid Pooling) as described in the paper.

ASPP applies on encoder outputs several parallel dilated convolutions with various dilation rates. This technique helps capturing longer range context without increasing too much the number of parameters.
The multi-scale design of the ASPP has proved to be receptive at the same time to details and greater contextual information.

So far, we have only considered Mobilenet V2 encoder.

## Network information


| Network Information     | Value                                                          |
|-------------------------|----------------------------------------------------------------|
|  Framework              | TensorFlow Lite                                                |
|  Quantization           | int8                                                           |
|  Provenance             | https://www.tensorflow.org/lite/examples/segmentation/overview |
|  Paper                  | https://arxiv.org/pdf/1706.05587                               |

The models are quantized using tensorflow lite converter.


## Network inputs / outputs


For an image resolution of NxM and P classes

| Input Shape  | Description |
|--------------| ----------- |
| (1, N, M, 3) | Single NxM RGB image with UINT8 values between 0 and 255 |

| Output Shape  | Description                                      |
|---------------|--------------------------------------------------|
| (1, N, M, 21) | Per-class confidence for P=21 classes in FLOAT32 |


Please refer to the stm32ai-modelzoo-services GitHub [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)
