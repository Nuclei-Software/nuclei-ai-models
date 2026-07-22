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

const int kKeypoints = 21;  // 21 hand landmarks (21 keypoints * 3 coords = 63)

// Dequantize int8 value to float using tensor's quantization parameters
float Dequantize(int8_t value, float scale, int32_t zero_point) {
  return static_cast<float>(value - zero_point) * scale;
}

// Get float value from tensor regardless of type (int8 quantized or float32)
float GetTensorFloatValue(TfLiteTensor* tensor, int index) {
  if (tensor->type == kTfLiteFloat32) {
    return tensor->data.f[index];
  } else if (tensor->type == kTfLiteInt8) {
    float scale = tensor->params.scale;
    int32_t zero_point = tensor->params.zero_point;
    return static_cast<float>(tensor->data.int8[index] - zero_point) * scale;
  } else if (tensor->type == kTfLiteUInt8) {
    float scale = tensor->params.scale;
    int32_t zero_point = tensor->params.zero_point;
    return static_cast<float>(tensor->data.uint8[index] - zero_point) * scale;
  }
  return 0.0f;
}

// hand_landmarks_postprocess: replicate the Python tflite_prediction.py post-processing logic
//
// Output tensors:
//   outputs[0]: htype    - hand type (right/left), shape [1, 1]
//   outputs[1]: norm_det - normalized detections, shape [1, 63]
//   outputs[2]: hprob    - hand presence probability, shape [1, 1]
//   outputs[3]: det      - raw detections, shape [1, 63]
//
// The post-processing aligns norm_det to the raw det space using per-axis
// (x, y, z) scale and offset computed via linear regression:
//   scale = cov(norm_axis, det_axis) / var(norm_axis)
//   offset = mean(det_axis) - scale * mean(norm_axis)
//   final_axis = norm_axis * scale + offset
void HandLandmarksPostprocess(TfLiteTensor* outputs[4], float* out_keypoints_x,
                               float* out_keypoints_y, float* out_keypoints_z,
                               float* out_htype, float* out_hprob) {
  // Read dequantized values for det (outputs[3]) and norm_det (outputs[1])
  float norm_det[63];
  float det[63];
  for (int i = 0; i < 63; i++) {
    norm_det[i] = GetTensorFloatValue(outputs[1], i);
    det[i] = GetTensorFloatValue(outputs[3], i);
  }

  // Compute means for x (offset 0), y (offset 1), z (offset 2) with stride 3
  float mean_nx = 0.0f, mean_ny = 0.0f, mean_nz = 0.0f;
  float mean_dx = 0.0f, mean_dy = 0.0f, mean_dz = 0.0f;
  for (int i = 0; i < kKeypoints; i++) {
    mean_nx += norm_det[i * 3 + 0];
    mean_ny += norm_det[i * 3 + 1];
    mean_nz += norm_det[i * 3 + 2];
    mean_dx += det[i * 3 + 0];
    mean_dy += det[i * 3 + 1];
    mean_dz += det[i * 3 + 2];
  }
  mean_nx /= kKeypoints;
  mean_ny /= kKeypoints;
  mean_nz /= kKeypoints;
  mean_dx /= kKeypoints;
  mean_dy /= kKeypoints;
  mean_dz /= kKeypoints;

  // Compute scale factors (linear regression slope: cov / var)
  // x_sc = Σ((nx - mean_nx) * (dx - mean_dx)) / Σ((nx - mean_nx)^2)
  float num_x = 0.0f, den_x = 0.0f;
  float num_y = 0.0f, den_y = 0.0f;
  float num_z = 0.0f, den_z = 0.0f;
  for (int i = 0; i < kKeypoints; i++) {
    float ndx = norm_det[i * 3 + 0] - mean_nx;
    float ndy = norm_det[i * 3 + 1] - mean_ny;
    float ndz = norm_det[i * 3 + 2] - mean_nz;
    float ddx = det[i * 3 + 0] - mean_dx;
    float ddy = det[i * 3 + 1] - mean_dy;
    float ddz = det[i * 3 + 2] - mean_dz;
    num_x += ndx * ddx;
    den_x += ndx * ndx;
    num_y += ndy * ddy;
    den_y += ndy * ndy;
    num_z += ndz * ddz;
    den_z += ndz * ndz;
  }
  // Avoid division by zero
  float x_sc = (den_x > 1e-6f) ? num_x / den_x : 1.0f;
  float y_sc = (den_y > 1e-6f) ? num_y / den_y : 1.0f;
  float z_sc = (den_z > 1e-6f) ? num_z / den_z : 1.0f;

  // Compute offset factors (replicating Python tflite_prediction.py logic)
  // Note: Python uses x_sc for ALL offset calculations (x_off, y_off, z_off)
  float x_off = mean_dx - x_sc * mean_nx;
  float y_off = mean_dy - x_sc * mean_ny;
  float z_off = mean_dz - x_sc * mean_nz;

  // Apply scaling and offset to norm_det to produce final keypoints
  // norm_det_scaled = norm_det * scale + offset
  for (int i = 0; i < kKeypoints; i++) {
    out_keypoints_x[i] = norm_det[i * 3 + 0] * x_sc + x_off;
    out_keypoints_y[i] = norm_det[i * 3 + 1] * y_sc + y_off;
    out_keypoints_z[i] = norm_det[i * 3 + 2] * z_sc + z_off;
  }

  // Read htype (outputs[0]) - hand type: near 0 → left, near 1 → right
  *out_htype = GetTensorFloatValue(outputs[0], 0);

  // Read hprob (outputs[2]) - hand presence probability
  *out_hprob = GetTensorFloatValue(outputs[2], 0);
}

void PrintResults(float* kpts_x, float* kpts_y, float* kpts_z,
                  float htype, float hprob) {
  printf("Hand presence probability: %.4f\n", hprob);
  printf("Hand type: %s (%.4f)\n", (htype > 0.5f) ? "right" : "left", htype);
  printf("Hand landmarks (21 keypoints in image space):\n");
  for (int i = 0; i < kKeypoints; i++) {
    printf("  %2d: (%.4f, %.4f, %.4f)\n", i, kpts_x[i], kpts_y[i], kpts_z[i]);
  }
}

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
 
  // Get image from provider. Model expects 224x224 RGB input.
  if (kTfLiteOk != GetImage(224, 224, 3, hand_landmark::input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != hand_landmark::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  // Get all 4 output tensors
  TfLiteTensor* outputs[4];
  for (int i = 0; i < 4; i++) {
    outputs[i] = hand_landmark::interpreter->output(i);
    if (outputs[i] == nullptr) {
      MicroPrintf("Output tensor %d is null.", i);
      return kTfLiteError;
    }
  }

  // Process the inference results.
  float kpts_x[21], kpts_y[21], kpts_z[21];
  float htype, hprob;
  HandLandmarksPostprocess(outputs, kpts_x, kpts_y, kpts_z, &htype, &hprob);
  PrintResults(kpts_x, kpts_y, kpts_z, htype, hprob);

  // DEBUG: Print raw det and norm_det values for comparison
  printf("\n===== C++ Raw Outputs (for debug comparison) =====\n");
  printf("det (outputs[3]):\n");
  for (int i = 0; i < 63; i++) {
    printf("%.6f ", GetTensorFloatValue(outputs[3], i));
    if ((i + 1) % 6 == 0) printf("\n");
  }
  printf("\nnorm_det (outputs[1]):\n");
  for (int i = 0; i < 63; i++) {
    printf("%.6f ", GetTensorFloatValue(outputs[1], i));
    if ((i + 1) % 6 == 0) printf("\n");
  }
  printf("=================================================\n");

  return kTfLiteOk;
}