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
#include "hand_landmarks_full_224_int8_pc_tflite.h"

namespace hand_landmark {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 2800 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  hand_landmark::model = tflite::GetModel(___pretrained_models_custom_dataset_hands_21kpts_hand_landmarks_full_224_int8_pc_tflite);
  if (hand_landmark::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        hand_landmark::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<8>  op_resolver;
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
  if (op_resolver.AddDequantize() != kTfLiteOk) {
      printf("Add Dequantize fail!\n");
      return;
  }
  if (op_resolver.AddLogistic() != kTfLiteOk) {
      printf("Add Logistic fail!\n");
      return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      hand_landmark::model, op_resolver, hand_landmark::tensor_arena, hand_landmark::kTensorArenaSize);
  hand_landmark::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = hand_landmark::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  hand_landmark::input = hand_landmark::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (hand_landmark::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

int main(int argc, char* argv[]) {
  setup();
 
  // Get image from provider.
  if (kTfLiteOk != GetImage(28, 28, 1, hand_landmark::input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != hand_landmark::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = hand_landmark::interpreter->output(0);

  // Process the inference results.
  float max = output->data.f[0];
  float score, precision = 0.0f;
  int32_t index = 0;
  for (int i = 0; i < 36; i++) {
    score = output->data.f[i];
    if (max < score) {
      max = score;
      index = i;
      precision = score;
    }
  }
  printf("Predict result is %c, precision: %f\r\n", class_names[index], precision);

  return kTfLiteOk;
}
