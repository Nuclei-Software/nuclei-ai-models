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

#include "microspeech_lstm_int8.h"

namespace lstm {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 300 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  lstm::model = tflite::GetModel(___pretrained_models_microspeech_lstm_microspeech_lstm_int8_tflite);
  if (lstm::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        lstm::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<4>  op_resolver;
  if (op_resolver.AddUnidirectionalSequenceLSTM() != kTfLiteOk) {
      printf("Add UnidirectionalSequenceLSTM fail!\n");
      return;
  }
  if (op_resolver.AddReshape() != kTfLiteOk) {
      printf("Add Reshape fail!\n");
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

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      lstm::model, op_resolver, lstm::tensor_arena, lstm::kTensorArenaSize);
  lstm::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = lstm::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  lstm::input = lstm::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (lstm::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}


extern char wavArray[12593];
unsigned char class_names[8][6] = {"right", "go", "no", "left", "stop", "up", "down", "yes"};
int main(int argc, char* argv[]) {
  setup();
 
  for (int i = 0; i < 49 * 257; i++) {
    int8_t* p_input = lstm::input->data.int8;
    p_input[i] = (int8_t)wavArray[i];

  }


  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != lstm::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* output = lstm::interpreter->output(0);
  // Process the inference results.
  int8_t max = output->data.int8[0];

  int32_t index = 0;
  for (int i = 0; i < 8; i++) {
    int8_t score = output->data.int8[i];
    if (max < score) {
      max = score;
      index = i;
    }
  }
  printf("Predict result is %s\r\n", class_names[index]);
  return kTfLiteOk;
}
