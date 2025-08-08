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

#include "wav2letter_int8.h"

namespace wav2letter {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 23612648;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  wav2letter::model = tflite::GetModel(___pretrained_models_wav2letter_wav2letter_int8_tflite);
  if (wav2letter::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        wav2letter::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<6>  op_resolver;
  if (op_resolver.AddReshape() != kTfLiteOk) {
      printf("Add Reshape fail!\n");
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
  if (op_resolver.AddSoftmax() != kTfLiteOk) {
      printf("Add Softmax fail!\n");
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
      wav2letter::model, op_resolver, wav2letter::tensor_arena, wav2letter::kTensorArenaSize);
  wav2letter::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = wav2letter::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  wav2letter::input = wav2letter::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (wav2letter::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

int ctc_greedy_decode(const float* logits, int T, int C, int seq_length, int** output) {
    if (seq_length > T) seq_length = T;
    if (seq_length <= 0) {
        *output = NULL;
        return 0;
    }

    // Step 1: argmax at each time step
    int* labels = (int*)malloc(seq_length * sizeof(int));
    for (int t = 0; t < seq_length; t++) {
        int best_class = 0;
        float best_score = logits[t * C + 0];
        for (int c = 1; c < C; c++) {
            float score = logits[t * C + c];
            if (score > best_score) {
                best_score = score;
                best_class = c;
            }
        }
        labels[t] = best_class;
    }

    // Step 2: merge repeated and remove blanks (blank_index = 0)
    int* decoded = (int*)malloc(seq_length * sizeof(int)); // worst case: no blanks, no repeats
    int decoded_len = 0;
    int prev_label = -1; // no previous

    for (int t = 0; t < seq_length; t++) {
        int current = labels[t];
        if (current == 28) { // char alphabet[] = {"abcdefghijklmnopqrstuvwxyz' @"};
            // blank, skip
            prev_label = -1; // optional: some implementations keep prev_label unchanged
            continue;
        }
        if (current != prev_label) {
            decoded[decoded_len++] = current;
        }
        prev_label = current;
    }

    free(labels);
    *output = decoded;
    return decoded_len;
}

extern signed char wavArray[40716];
char alphabet[] = {"abcdefghijklmnopqrstuvwxyz' @"};
float cur_output_data[522 * 29];
float* p_cur_output_data = cur_output_data;
char Transcribed_File[300];
int main(int argc, char* argv[]) {
  setup();

  int32_t context = 24 + 2 * (7 * 3 + 16);  // = 98 - theoretical max receptive field on each side
  int32_t size =  296;                      // input_details['shape'][1]
  int32_t inner = size - 2 * context;       // 100
  int32_t data_end = 1044;                  // data.shape[1]

  int32_t start,end,y_start,y_end,shift;

  int32_t data_pos = 0;
  while (data_pos < data_end) {
    if (data_pos == 0) {
        // Align inputs from the first window to the start of the data and include the intial context in the output
        start = data_pos;
        end = start + size;
        y_start = 0;
        y_end = y_start + (size - context) / 2;
        data_pos = end - context;
    } else if (data_pos + inner + context >= data_end) {
        // Shift left to align final window to the end of the data and include the final context in the output
        shift = (data_pos + inner + context) - data_end;
        start = data_pos - context - shift;
        end = start + size;
        assert(start >= 0);
        y_start = (shift + context) / 2;  // Will be even because we assert it above
        y_end = size / 2;
        data_pos = data_end;
    } else {
        // Capture only the inner region from mid-input inferences, excluding output from both context regions
        start = data_pos - context;
        end = start + size;
        y_start = context / 2;
        y_end = y_start + inner / 2;
        data_pos = end - context;
    }

    int8_t* p_input = wav2letter::input->data.int8;
    for (int i = start; i < end; i++) {
      for (int j = 0; j < 39; j++) {
          *p_input++ = (int8_t)wavArray[i * 39 + j];
      }
    }
    // Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != wav2letter::interpreter->Invoke()) {
      MicroPrintf("Invoke failed.");
    }
    TfLiteTensor* output = wav2letter::interpreter->output(0);
    int8_t* p_output = output->data.int8;

    for (int i = y_start; i < y_end; i++) {
      for (int j = 0; j < 29; j++) {
        *p_cur_output_data++ = ((float)(p_output[i * 29 + j]) + 128) * 0.00390625;
      }
    }
  }

  int* result;
  int decoded_len = ctc_greedy_decode(cur_output_data, 522, 29, 522, &result);

  for (int i = 0; i < decoded_len - 1; i++) {
    Transcribed_File[i] = alphabet[result[i]];
  }
  Transcribed_File[decoded_len] = '\0';
  printf("Transcribed File: %s\r\n", Transcribed_File);

  return kTfLiteOk;
}
