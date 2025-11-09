# Nuclei AI Models

[中文](README_zh.md) \| English

The **Nuclei AI Models** repository collects dozens of TFLite models. Combined with `nuclei-sdk`, users can quickly deploy and run these models on Nuclei Cores or Nuclei QEMU, and observe their performance.

---

### **Important Notes:**

* Model training and quantization are not included. Users must either train and quantize models themselves, or download existing TFLite models.
* Currently, only **int8** and **int16** TFLite models are supported.

---

### **Supported Models:**

* Image Classification (image_classification)
* Object Detection (object_detection)
* License Plate Recognition (LPRNet)
* Pose Estimation (Gesture Detection)
* Keyword Recognition (audio)
* Instance Segmentation (instance_segmentation)
* Semantic Segmentation (semantic_segmentation)   
...

---

## 1. How to Use This Repository?

Follow the steps below:

### **Step 1: Clone the Repository**

```sh
$ git clone https://github.com/Nuclei-Software/nuclei-ai-models
```

### **Step 2: Download Toolchain and Dependencies**

Run the setup script to automatically download required dependencies and configure the toolchain path:

```sh
$ cd nuclei-ai-models
$ source setup.sh
```

This script performs the following tasks:

1. Downloads [nuclei_studio](https://download.nucleisys.com/upload/files/nucleistudio/NucleiStudio_IDE_202502-lin64.tgz) and [nuclei-sdk](https://github.com/Nuclei-Software/nuclei-sdk/archive/refs/tags/0.8.0.zip) into the `downloads` folder.
2. Downloads [tflm](https://github.com/Nuclei-Software/npk-tflm/archive/refs/tags/0.5.0.zip) into `downloads/nuclei-sdk/Components`.
3. Sets the toolchain environment path.

### **Step 3: Run a Demo**

Take the **MNIST** model as an example, running it on **Nuclei QEMU**:

```sh
$ cd modelzoo/image_classification/mnist/src
$ bash prepare.sh
$ make SOC=evalsoc CORE=n300fd clean
$ make SOC=evalsoc CORE=n300fd run_qemu
```

Sample log output (Predict num is 7, indicating successful execution):

```log
Run program mnist.elf on qemu-system-riscv32
qemu-system-riscv32 -M nuclei_evalsoc,download=ddr -cpu nuclei-n300fd,ext= -smp 1 -icount shift=0 -nodefaults -nographic -serial stdio -kernel mnist.elf
Nuclei SDK Build Time: Aug 13 2025, 11:04:46
Download Mode: DDR
CPU Frequency 1000005632 Hz
CPU HartID: 0
Predict result is 7, precision: 0.988281
```

---

## 2. How to Add a New Use Case?

Users can train and quantize a new int8/int16 TFLite model and deploy it in this repository. Follow the steps below:

### **Step 1: Obtain a TFLite Model**

For example, using the MNIST model, you have trained and quantized a model named: `mnist_int8.tflite`.

Alternatively, download a pre-quantized MNIST model from the web.

### **Step 2: Create a New Use Case**

Following the structure of existing examples, create a new MNIST demo under the `modelzoo` directory.

The directory structure should be:

```sh
pretrained_models     # Store TFLite models
script                # Python inference scripts
src                   # Embedded/MCU-side inference code
README.md             # Description of the use case
```

### **Step 3: Evaluate the Model**

Place the `mnist_int8.tflite` model into the `pretrained_models` folder. Write a Python script in the `script` folder to run inference using TensorFlow, so you can verify the model's performance on the host side.

First, install required dependencies:

```sh
$ cd nuclei-ai-models
$ python -m pip install -r requirement.txt
```

Run inference on the host:

```sh
$ cd mnist/script
$ python tflite_prediction.py
```

Host-side log output:

```log
img.shape is (28, 28)
x_test.shape is (1, 28, 28, 1)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
The result is: 7 Probability: 0.98828125
```

### **Step 4: Serialize the Model and Test Image**

Use the Linux `xxd` tool to convert the model into a hexadecimal C array:

```sh
$ xxd -i mnist_int8.tflite > mnist_int8.cc
```

This generates a C array like:

```c
unsigned char ___pretrained_models_mnist_int8_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x00, 0x00, 0xb0, 0x00, 0x00, 0x00, 0x58, 0x2a, 0x00, 0x00,
  0x68, 0x2a, 0x00, 0x00, 0x10, 0x4d, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  ...
};
unsigned int ___pretrained_models_mnist_int8_tflite_len = 19944;
```

Similarly, serialize the test image for inference:

```sh
$ cd mnist/script
$ python im2array.py
```

This generates `test_image_provider.cc`.

### **Step 5: Write the Inference Code**

Copy the generated `mnist_int8.cc` and `test_image_provider.cc` into the `src` folder. Write the inference code by referring to existing examples.

To register required operators, use the provided Python script for assistance:

```sh
$ python /path/to/nuclei-ai-models/common/scripts/generate_op_resolver_code.py mnist_int8.tflite
```

The script generates code like:

```c
Operator number is 7
if (op_resolver.AddQuantize() != kTfLiteOk) {
    printf("Add Quantize fail!\n");
    return;
}
if (op_resolver.AddConv2D() != kTfLiteOk) {
    printf("Add Conv2D fail!\n");
    return;
}
if (op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
    printf("Add DepthwiseConv2D fail!\n");
    return;
}
if (op_resolver.AddMean() != kTfLiteOk) {
    printf("Add Mean fail!\n");
    return;
}
if (op_resolver.AddFullyConnected() != kTfLiteOk) {
    printf("Add FullyConnected fail!\n");
    return;
}
if (op_resolver.AddSoftmax() != kTfLiteOk) {
    printf("Add Softmax fail!\n");
    return;
}
if (op_resolver.AddDequantize() != kTfLiteOk) {
    printf("Add Dequantize fail!\n");
    return;
}
```

You can copy and paste this into your inference code.

### **Step 6: Write the Makefile**

After writing the inference code, create a `Makefile`. You can refer to existing examples.

Example `Makefile`:

```makefile
TARGET = mnist

AI_MODELS_ROOT := ../../../../

# REBUILD_TFLM = 0: Use pre-built TFLM static library (supported for n300fd and nx900fd)
# REBUILD_TFLM = 1: Compile TFLM from source (slower)
REBUILD_TFLM ?= 1

CORE ?= n300fd

ARCH_EXT ?= v

DOWNLOAD ?= ddr

SRCDIRS = .

INCDIRS = .

NMSIS_LIB := nmsis_nn

STDCLIB ?= newlib_small

COMMON_FLAGS := -O3

include $(AI_MODELS_ROOT)/modelzoo/Makefile.common
```

> **Note:**  
> Since compiling TFLM from source is time-consuming, the `REBUILD_TFLM` flag allows you to choose between using a static library or building from source.  
> If `CORE` is set to `n300fd` or `nx900fd` and no additional extensions are enabled (i.e., `ARCH_EXT` is empty), you can set `REBUILD_TFLM ?= 0` to use the static library and reduce build time. Otherwise, set `REBUILD_TFLM ?= 1`.

### **Step 7: Compile and Run the Demo**

You can run the demo on a **Nuclei Core** or **Nuclei QEMU**. Here is an example using QEMU:

```sh
$ cd modelzoo/image_classification/mnist/src
$ make SOC=evalsoc CORE=n300fd clean
$ make SOC=evalsoc CORE=n300fd run_qemu
```

---

### **References:**

1. [Arm Model Zoo](https://github.com/ARM-software/ML-zoo/)
2. [STM32AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
3. [STM32AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)
4. [STM32AI Ultralytics](https://github.com/stm32-hotspot/ultralytics)
5. [PINTO Model Zoo](https://github.com/PINTO0309/PINTO_model_zoo)
6. [TFLite-Micro-Seq2Seq](https://github.com/da03/TFLite-Micro-Seq2Seq.git)
7. [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
8. [awesome-tensorflow-lite](https://github.com/margaretmz/awesome-tensorflow-lite)
