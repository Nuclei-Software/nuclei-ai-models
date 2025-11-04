/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <math.h>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "image_provider.h"
#include "mobilenet_v2_0_35_224_int8_tflite.h"

namespace mobilenet_v2 {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 1694264;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  mobilenet_v2::model = tflite::GetModel(___pretrained_models_ImageNet_mobilenet_v2_0_35_224_mobilenet_v2_0_35_224_int8_tflite);
  if (mobilenet_v2::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        mobilenet_v2::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<9>  op_resolver;
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
  if (op_resolver.AddPad() != kTfLiteOk) {
      printf("Add Pad fail!\n");
      return;
  }
  if (op_resolver.AddAdd() != kTfLiteOk) {
      printf("Add Add fail!\n");
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

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      mobilenet_v2::model, op_resolver, mobilenet_v2::tensor_arena, mobilenet_v2::kTensorArenaSize);
  mobilenet_v2::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = mobilenet_v2::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  mobilenet_v2::input = mobilenet_v2::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (mobilenet_v2::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

typedef struct {
  float precision;
  int index;
} Element;

void swap(Element *a, Element *b) {
  Element temp = *a;
  *a = *b;
  *b = temp;
}

static void Heapify(Element arr[], int len, int idx) {
  int child = idx * 2 + 1;
  while (child < len) {
      if (child + 1 < len && arr[child + 1].precision < arr[child].precision) {
          ++child;
      }

      if (arr[child].precision < arr[idx].precision) {
          swap(&arr[child], &arr[idx]);
          idx = child;
          child = idx * 2 + 1;
      } else {
          break;
      }
  }
}

void sortDescending(Element topk[], int k) {
  for (int i = 0; i < k - 1; ++i) {
      for (int j = 0; j < k - i - 1; ++j) {
          if (topk[j].precision < topk[j + 1].precision) {
              swap(&topk[j], &topk[j + 1]);
          }
      }
  }
}

void findTopK(float array[], int n, int k, Element topk[]) {
  if (k <= 0 || k > n) {
      printf("Invalid value of k.\n");
      return;
  }

  Element elements[k];
  for (int i = 0; i < k; ++i) {
      elements[i].precision = array[i];
      elements[i].index = i;
  }

  for (int i = k / 2 - 1; i >= 0; --i) {
      Heapify(elements, k, i);
  }

  for (int i = k; i < n; ++i) {
      if (array[i] > elements[0].precision) {
          elements[0].precision = array[i];
          elements[0].index = i;
          Heapify(elements, k, 0);
      }
  }

  for (int i = 0; i < k; ++i) {
      topk[i] = elements[i];
  }

  sortDescending(topk, k);
}

extern const char *class_names[1000];
int main(int argc, char* argv[]) {
  setup();
 
  // Get image from provider.
  if (kTfLiteOk != GetImage(224, 224, 3, mobilenet_v2::input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != mobilenet_v2::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = mobilenet_v2::interpreter->output(0);

  // Process the inference results.
  float *output_array = output->data.f;
  int output_array_len = 1000;
  int k = 3; // Top-3
  Element topk[k];
  findTopK(output_array, output_array_len, k, topk);
  for (int i = 0; i < k; ++i) {
      printf("Top%d is %s, precision: %f\n", i, class_names[topk[i].index], topk[i].precision);
  }

  return kTfLiteOk;
}
