
# MNIST

## Use case : Image classification

## Model description

**Model description:**
mnist lstm means a deep learning model that uses LSTM (Long Short-Term Memory) networks to classify the MNIST handwritten digit dataset. It is int8 tflite framework.


**Network inputs / outputs:**


For an image resolution of 28x28 and 10 classes : 10 integers

| Input Shape | Description |
| ----- | ----------- |
| (1, 28, 28) | int8 |

| Output Shape | Description |
| ----- | ----------- |
| (1, 10) | Per-class confidence for 10 classes in int8 |

## DataSet

Dataset details: [link](https://www.nist.gov/itl/products-and-services/emnist-dataset)

