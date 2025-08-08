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
#include "ocr_model_quant_int8.h"
#include "image_provider.h"

namespace lprnet {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 1266753;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  lprnet::model = tflite::GetModel(___pretrained_models_ocr_model_quant_int8_tflite);
  if (lprnet::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        lprnet::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<4>  op_resolver;
  if (op_resolver.AddConv2D() != kTfLiteOk) {
      printf("Add Conv2D fail!\n");
      return;
  }
  if (op_resolver.AddMaxPool2D() != kTfLiteOk) {
      printf("Add MaxPool2D fail!\n");
      return;
  }
  if (op_resolver.AddReshape() != kTfLiteOk) {
      printf("Add Reshape fail!\n");
      return;
  }
  if (op_resolver.AddQuantize() != kTfLiteOk) {
      printf("Add Quantize fail!\n");
      return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      lprnet::model, op_resolver, lprnet::tensor_arena, lprnet::kTensorArenaSize);
  lprnet::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = lprnet::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  lprnet::input = lprnet::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (lprnet::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

static inline uint16_t argmax(uint8_t *data, uint16_t length)
{
  uint8_t max = data[0];
  uint16_t index = 0;
  
  for (int i = 1; i < length; i++) {
    if (data[i] > max) {
      max = data[i];
      index = i;
    }
  }
  return index;
}

const char chars[84][6] = {"jing", "hu", "jin", "yu", "ji", "ji", "meng", "liao", "ji", "hei", "su", "zhe", "wan", "min", "gan", "lu", "yu", "e", "xiang", "yue", "gui",
         "qiong", "chuan", "gui", "yun", "zang", "shan", "gan", "qing", "ning", "xin", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z","gang","xue","shi","jing","ao","gua","jun","bei","nan","guang","shen","lan","cheng","ji","hai","ming","hang","kong"
};

int main(int argc, char* argv[]) {
  setup();
  int H, W, C;

  H = 160;
  W = 40;
  C = 3;

  // Get image from provider.
  if (kTfLiteOk != GetImage(H, W, C, lprnet::input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != lprnet::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* outp = lprnet::interpreter->output(0);
 
  uint8_t *output = outp->data.uint8;

  unsigned char res[50] = {'\0'};
  uint16_t len = 0;
  int max_idx = 0;
  int last_max_idx = -1;
  for (int i = 2; i < 16; i++) {
        max_idx = argmax(&output[i * 84], 84);
        if (max_idx == last_max_idx) {
          continue;
        } else {
          last_max_idx = max_idx;
        }
        strcpy(res + len, chars[max_idx]);
        len += strlen(chars[max_idx]);
  }
  
  printf("result is %s \r\n", res);

  return kTfLiteOk;

}
