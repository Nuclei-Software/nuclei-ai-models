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

#include "ad01_int8_tflite.h"

float compute_y_pred(const float* data_fp, const float* out, int rows, int cols) {
    float sum_errors = 0.0f;

    for (int i = 0; i < rows; i++) {
        float sum_sq_diff = 0.0f;
        for (int j = 0; j < cols; j++) {
            float diff = data_fp[i * cols + j] - out[i * cols + j];
            sum_sq_diff += diff * diff;
        }
        float mse = sum_sq_diff / (float)cols;  // mean over cols (axis=1)
        sum_errors += mse;
    }

    return sum_errors / (float)rows;  // mean of errors
}

namespace deep_autoencoder {
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
  deep_autoencoder::model = tflite::GetModel(___pretrained_models_deep_autoencoder_ad01_int8_tflite);
  if (deep_autoencoder::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        deep_autoencoder::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<1>  op_resolver;
  if (op_resolver.AddFullyConnected() != kTfLiteOk) {
    printf("Add FullyConnected fail!\n");
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      deep_autoencoder::model, op_resolver, deep_autoencoder::tensor_arena, deep_autoencoder::kTensorArenaSize);
  deep_autoencoder::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = deep_autoencoder::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  deep_autoencoder::input = deep_autoencoder::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (deep_autoencoder::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}
extern float wavArray[125440];
float outputArray[125440]; 
int main(int argc, char* argv[]) {
  setup();
 
  for (int i = 0; i < 196; i++) {
    int8_t* p_input = deep_autoencoder::input->data.int8;
    for (int j = 0; j < 640; j++) {
      p_input[j] = (int8_t)(wavArray[i * 640 + j] / 0.3910152316093445 + 89);

      // Run the model on this input and make sure it succeeds.
      if (kTfLiteOk != deep_autoencoder::interpreter->Invoke()) {
        MicroPrintf("Invoke failed.");
      }

      TfLiteTensor* output = deep_autoencoder::interpreter->output(0);
      int8_t* p_output = output->data.int8;
      outputArray[i * 640 + j] = ((float)(p_output[j]) - 96) * 0.36449846625328064;
    }
  }

  float error = compute_y_pred(wavArray, outputArray, 196, 640);

  printf("Prediction %f\r\n", error);
  return kTfLiteOk;
}
