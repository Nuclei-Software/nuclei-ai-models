# ResNet50 v2

## **Use case** : `Image classification`

# Model description


ResNets family is a well known architecture that uses skip connections to enable stronger gradients in much deeper networks. This variant has 50 layers.

The model is quantized in int8 using tensorflow lite converter.

## Network information


| Network Information     |  Value          |
|-------------------------|-----------------|
|  Framework              | TensorFlow Lite |
|  MParams                | 25.6 M          |
|  Quantization           | int8            |
|  Provenance             | https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50V2 |
|  Paper                  | https://arxiv.org/abs/1603.05027 |

The models are quantized using tensorflow lite converter.


## Network inputs / outputs


For an image resolution of NxM and P classes

| Input Shape | Description |
| ----- | ----------- |
| (1, N, M, 3) | Single NxM RGB image with UINT8 values between 0 and 255 |

| Output Shape | Description |
| ----- | ----------- |
| (1, P) | Per-class confidence for P classes in FLOAT32|



# Performances

## Metrics

- Measures are done with default STM32Cube.AI configuration with enabled input / output allocated option.
- `tfs` stands for "training from scratch", meaning that the model weights were randomly initialized before training.
- `tl` stands for "transfer learning", meaning that the model backbone weights were initialized from a pre-trained model, then only the last layer was unfrozen during the training.
- `fft` stands for "full fine-tuning", meaning that the full model weights were initialized from a transfer learning pre-trained model, and all the layers were unfrozen during the training.




### Accuracy with ImageNet dataset

Dataset details: [link](https://www.image-net.org), Quotation[[4]](#4).
Number of classes: 1000.
To perform the quantization, we calibrated the activations with a random subset of the training set.
For the sake of simplicity, the accuracy reported here was estimated on the 50000 labelled images of the validation set.

|model    | Format | Resolution | Top 1 Accuracy |
|---------|--------|------------|----------------|
| [ResNet50 v2 ](./Public_pretrainedmodel_public_dataset/ImageNet/resnet50_v2_224/resnet50_v2_224_int8.tflite) | Int8 | 224x224x3    | 65.99 % |

## Retraining and Integration in a simple example:

Please refer to the stm32ai-modelzoo-services GitHub [here](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)



# References

<a id="1">[1]</a>
L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101 -- Mining Discriminative Components with Random Forests." European Conference on Computer Vision, 2014.