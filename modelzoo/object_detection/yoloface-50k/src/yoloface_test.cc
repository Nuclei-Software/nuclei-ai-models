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
#include "yoloface_int8_tflite.h"
#include "image_provider.h"

namespace yoloface {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 136 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  yoloface::model = tflite::GetModel(___pretrained_models_yoloface_int8_tflite);
  if (yoloface::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        yoloface::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<9>  op_resolver;
  if (op_resolver.AddPad() != kTfLiteOk) {
    printf("Add Pad fail!\n");
    return;
  }
  if (op_resolver.AddConv2D() != kTfLiteOk) {
      printf("Add Conv2D fail!\n");
      return;
  }
  if (op_resolver.AddLeakyRelu() != kTfLiteOk) {
      printf("Add LeakyRelu fail!\n");
      return;
  }
  if (op_resolver.AddDepthwiseConv2D() != kTfLiteOk) {
      printf("Add DepthwiseConv2D fail!\n");
      return;
  }
  if (op_resolver.AddMaxPool2D() != kTfLiteOk) {
      printf("Add MaxPool2D fail!\n");
      return;
  }
  if (op_resolver.AddAdd() != kTfLiteOk) {
      printf("Add Add fail!\n");
      return;
  }
  if (op_resolver.AddConcatenation() != kTfLiteOk) {
      printf("Add Concatenation fail!\n");
      return;
  }
  if (op_resolver.AddQuantize() != kTfLiteOk) {
      printf("Add Quantize fail!\n");
      return;
  }
  if (op_resolver.AddDequantize() != kTfLiteOk) {
      printf("Add Dequantize fail!\n");
      return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      yoloface::model, op_resolver, yoloface::tensor_arena, yoloface::kTensorArenaSize);
  yoloface::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = yoloface::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  yoloface::input = yoloface::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (yoloface::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

static inline float sigmod(float x)
{
  float y = 1/(1+expf(-x));
  return y;
}

uint8_t anchors[3][2] = {{9, 14}, {12, 17}, {22, 21}};
int main(int argc, char* argv[]) {
   setup();

   float w_scale = 1.0;
   float h_scale = 1.0;
   int H, W, C;

   H = 440;
   W = 329;
   C = 3;

   w_scale = W * 1.0 / 56;
   h_scale = H * 1.0 / 56;

  // Get image from provider.
  if (kTfLiteOk != GetImage(59, 59, 3, yoloface::input->data.int8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != yoloface::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* outp = yoloface::interpreter->output(0);
 
  // Initialize output tensor with values
  int8_t *out_data = outp->data.int8;
  int grid_x, grid_y;
  float x, y, w, h;
  int x_min, x_max, y_min, y_max;
  for (int i = 0; i < 7 * 7; i++) {
    for (int j = 0; j < 3; j++) {
      int8_t conf = out_data[i * 18 + j * 6 + 4];
      float pred = sigmod((conf + 15) * 0.14218327403068542f);
      // nms
      if (pred > 0.7) {
        x = ((float)out_data[i * 18 + j * 6] + 15) * 0.14218327403068542f;
        y = ((float)out_data[i * 18 + j * 6 + 1] + 15) * 0.14218327403068542f;
        w = ((float)out_data[i * 18 + j * 6 + 2] + 15) * 0.14218327403068542f;
        h = ((float)out_data[i * 18 + j * 6 + 3] + 15) * 0.14218327403068542f;
        grid_x = i % 7;
        grid_y = i / 7;
        x = (sigmod(x) + grid_x) * 8;
        y = (sigmod(y) + grid_y) * 8;
        w = expf(w) * anchors[j][0];
        h = expf(h) * anchors[j][1];

        x_min = (x - w / 2) * w_scale;
        x_max = (x + w / 2) * w_scale;
        y_min = (y - h / 2) * h_scale;
        y_max = (y + h / 2) * h_scale;

        printf("x_min: %d, y_min: %d, x_max: %d, y_max: %d, prediction = %0.2f\n",
        x_min, y_min, x_max, y_max, pred);

      }
    }
  }
  return kTfLiteOk;
}
