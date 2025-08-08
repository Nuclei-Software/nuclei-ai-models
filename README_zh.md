# Nuclei AI Models

中文 \| [English](README.md)

Nuclei AI models 仓库收集了数十个Tflite模型，结合nuclei-sdk，用户可快速在Nuclei Core上或Nuclei QEMU 上运行这些模型，并观察模型运行效果。

**注意事项：**

* 不包含模型训练及量化，用户需自己训练并量化模型，或者下载已有的tflite格式模型
* 目前只支持int8/int16 类型的tflite模型

**支持的模型：**

* 图像分类（image_classification）
* 图像检测 (object_detection)  
* 车牌检测（LPRNet）
* 手势检测（pose_estimation）
* 关键词识别 (audio)
* 实例分割 (instance_segmentation)
* 语义分割 (semantic_segmentation)  
...

## 1 如何使用这个仓库？

步骤如下：

**step1: 下载仓库**

~~~sh
$ git clone https://github.com/Nuclei-Software/nuclei-ai-models
~~~

**step2: 下载工具链以及其他依赖**

执行脚本，会自动下载所需依赖，并设置工具链路径

~~~sh
$ cd nuclei-ai-models
$ source setup.sh
~~~

脚本会完成如下操作：

1. 下载[nuclei_studio](https://download.nucleisys.com/upload/files/nucleistudio/NucleiStudio_IDE_202502-lin64.tgz)，[nuclei-sdk](https://github.com/Nuclei-Software/nuclei-sdk/archive/refs/tags/0.8.0.zip) 到downloads文件夹下
2. 下载[tflm](https://github.com/Nuclei-Software/npk-tflm/archive/refs/tags/0.5.0.zip)到downloads/nuclei-sdk/Components 下
3.  设置工具链路径。

**step3: 运行用例**

以mnist模型为例，运行到nuclei qemu上：

~~~sh
$ cd modelzoo/image_classification/mnist/src
$ make SOC=evalsoc CORE=n300fd clean
$ make SOC=evalsoc CORE=n300fd run_qemu
~~~

其日志输出如下，识别数字为7，可见运行正常：

~~~log
Run program mnist.elf on qemu-system-riscv32
qemu-system-riscv32 -M nuclei_evalsoc,download=ddr -cpu nuclei-n300fd,ext= -smp 1 -icount shift=0 -nodefaults -nographic -serial stdio -kernel mnist.elf
Nuclei SDK Build Time: Aug 13 2025, 11:04:46
Download Mode: DDR
CPU Frequency 1000005632 Hz
CPU HartID: 0
Predict result is 7, precision: 0.988281
~~~

## 2 如何增加新用例？

用户训练量化得到新的int8/int16 tflite模型，也可以放到此仓库进行部署运行，其步骤如下：

**step1: 获取tflite模型**

以mnist模型为例，我们训练量化得到了一个mnist模型：`mnist_int8.tflite`

或者，从网上下载一个已经量化好的mnist模型

**step2: 新建用例**

模仿其它用例结构，在modelzoo文件夹下新建mnist用例。

mnist用例的目录结构如下：

~~~sh
pretrained_models  # 存放tflite模型
script             # 存放tflite python推理脚本
src                # 嵌入式MCU端推理代码
README.md          # 用例说明
~~~

**step3: 评估模型**

将 mnist_int8.tflite 模型放置到 pretrained_models 文件夹，在script文件夹下编写python脚本，调用Tensorflow模块，运行推理。这样我们可以在主机端查看模型运行效果。

需要先安装Tensorflow等依赖

~~~sh
$ cd nuclei-ai-models
$ python -m pip install -r requirement.txt
~~~

在主机侧运行推理：

~~~sh
$ cd mnist/script
$ python tflite_prediction.py
~~~
主机侧log如下:
~~~log
img.shape is (28 28)
x_test.shape is (1, 28, 28, 1)
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
The result is: 7 Probability:0.98828125
~~~

**step4: 将模型以及测试图像序列化**

使用linux的xxd工具可以将模型转化为16进制数组

~~~sh
$ xxd -i mnist_int8.tflite > mnist_int8.cc
~~~

得到了模型的16进制数组，如下：

~~~c
unsigned char ___pretrained_models_mnist_int8_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, 0x14, 0x00, 0x20, 0x00,
  0x1c, 0x00, 0x18, 0x00, 0x14, 0x00, 0x10, 0x00, 0x0c, 0x00, 0x00, 0x00,
  0x08, 0x00, 0x04, 0x00, 0x14, 0x00, 0x00, 0x00, 0x1c, 0x00, 0x00, 0x00,
  0x80, 0x00, 0x00, 0x00, 0xb0, 0x00, 0x00, 0x00, 0x58, 0x2a, 0x00, 0x00,
  0x68, 0x2a, 0x00, 0x00, 0x10, 0x4d, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
  ...
}
unsigned int ___pretrained_models_mnist_int8_tflite_len = 19944;
~~~

同样的，我们需要对待识别的数字图片进行序列化：

~~~sh
$ cd mnist/script
$ python im2array.py
~~~

得到 test_image_provider.cc

**step5: 编写测试用例**

将得到的模型文件mnist_int8.cc 以及待识别图片数据 test_image_provider.cc 拷贝到src文件夹，可模仿其它用例，编写推理代码。

在注册算子时，可以使用python脚本辅助完成：

~~~sh
$ python /path/to/nuclei-ai-models/common/scripts/generate_op_resolver_code.py mnist_int8.tflite
~~~

会生成如下代码，可以粘贴到推理代码对应位置：

~~~c
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
~~~

**step6: 编写Makefile**

推理代码编写完成后，需要编写Makefile，同样的，可以参考其它用例的写法。

~~~sh
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
~~~

**注意：**

由于tflm组件编译比较耗时，所以在Makefile中提供REBUILD_TFLM 宏来选择使用静态库还是编译tflm源码

如果CORE设置n300fd或者nx900fd，并且不开其它扩展（也即ARCH_EXT为空），则可以设置REBUILD_TFLM ?= 0 来选择tflm静态库，从而减少编译时间，否则需要设置REBUILD_TFLM ?= 1

**step7: 编译并运行用例**

可以运行在Nuclei Core上或qemu上，这里以运行在Nuclei qemu上为例

~~~sh
$ cd modelzoo/image_classification/mnist/src
$ make SOC=evalsoc CORE=n300fd clean
$ make SOC=evalsoc CORE=n300fd run_qemu
~~~



**参考：**

1. [Arm Model Zoo](https://github.com/ARM-software/ML-zoo/)
2. [STM32AI Model Zoo](https://github.com/STMicroelectronics/stm32ai-modelzoo)
3. [STM32AI Model Zoo Services](https://github.com/STMicroelectronics/stm32ai-modelzoo-services)
4. [STM32AI Ultralytics](https://github.com/stm32-hotspot/ultralytics)
5. [PINTO Model Zoo](https://github.com/PINTO0309/PINTO_model_zoo)
6. [TFLite-Micro-Seq2Seq](https://github.com/da03/TFLite-Micro-Seq2Seq.git)
7. [RKNN Model Zoo](https://github.com/airockchip/rknn_model_zoo)
8. [awesome-tensorflow-lite](https://github.com/margaretmz/awesome-tensorflow-lite)

