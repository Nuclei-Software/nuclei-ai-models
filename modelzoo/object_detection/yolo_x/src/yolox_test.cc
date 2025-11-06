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
#include "yolo_x_nano_256_0_5_0_4_int8_tflite.h"
#include "image_provider.h"

namespace tinyYoloV2 {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 3 * 1024 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup() {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  tinyYoloV2::model = tflite::GetModel(___pretrained_models_coco_2017_person_yolo_x_nano_256_yolo_x_nano_256_0_5_0_4_int8_tflite);
  if (tinyYoloV2::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        tinyYoloV2::model->version(), TFLITE_SCHEMA_VERSION);
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
  if (op_resolver.AddConcatenation() != kTfLiteOk) {
      printf("Add Concatenation fail!\n");
      return;
  }
  if (op_resolver.AddMaxPool2D() != kTfLiteOk) {
      printf("Add MaxPool2D fail!\n");
      return;
  }
  if (op_resolver.AddResizeNearestNeighbor() != kTfLiteOk) {
      printf("Add ResizeNearestNeighbor fail!\n");
      return;
  }
  if (op_resolver.AddDequantize() != kTfLiteOk) {
      printf("Add Dequantize fail!\n");
      return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      tinyYoloV2::model, op_resolver, tinyYoloV2::tensor_arena, tinyYoloV2::kTensorArenaSize);
  tinyYoloV2::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = tinyYoloV2::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  tinyYoloV2::input = tinyYoloV2::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (tinyYoloV2::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int x_min, y_min; // 左上角坐标
    int x_max, y_max; // 右下角坐标
    float score;      // 置信度分数
} Bbox;

Bbox boxes[128];

float iou(Bbox a, Bbox b) {
    int inter_x_min = MAX(a.x_min, b.x_min);
    int inter_y_min = MAX(a.y_min, b.y_min);
    int inter_x_max = MIN(a.x_max, b.x_max);
    int inter_y_max = MIN(a.y_max, b.y_max);

    int inter_area = MAX(inter_x_max - inter_x_min + 1, 0.0f) * MAX(inter_y_max - inter_y_min + 1, 0.0f);
    int box_a_area = (a.x_max - a.x_min + 1) * (a.y_max - a.y_min + 1);
    int box_b_area = (b.x_max - b.x_min + 1) * (b.y_max - b.y_min + 1);

    return inter_area * 1.0 / (box_a_area + box_b_area - inter_area);
}

// 用于交换两个Bbox
void swap_bbox(Bbox* a, Bbox* b) {
    Bbox temp = *a;
    *a = *b;
    *b = temp;
}

// qsort 比较函数
int compare(const void* a, const void* b) {
    Bbox bbox_a = *(Bbox*)a;
    Bbox bbox_b = *(Bbox*)b;
    if (bbox_a.score > bbox_b.score) return -1;
    else if (bbox_a.score < bbox_b.score) return 1;
    else return 0;
}

void nms(Bbox* boxes, int num_boxes, float iou_threshold, Bbox** out_boxes, int* out_num_boxes) {
    qsort(boxes, num_boxes, sizeof(Bbox), compare);

    int selected[num_boxes];
    int count = 0;

    for (int i = 0; i < num_boxes; ++i) {
        int keep = 1;
        for (int j = 0; j < count; ++j) {
            if (iou(boxes[i], boxes[selected[j]]) > iou_threshold) {
                keep = 0;
                break;
            }
        }
        if (keep) {
            selected[count++] = i;
        }
    }

    *out_boxes = (Bbox*)malloc(count * sizeof(Bbox));
    for (int i = 0; i < count; ++i) {
        (*out_boxes)[i] = boxes[selected[i]];
    }
    *out_num_boxes = count;
}

static inline float sigmod(float x)
{
  float y = 1/(1+expf(-x));
  return y;
}

float anchors[2] = {0.5, 0.5};

int main(int argc, char* argv[]) {
   setup();

   float w_scale = 1.0;
   float h_scale = 1.0;
   int H = 256, W = 256, C = 3;
   int num_boxes = 0;
   Bbox* out_boxes;
   int out_num_boxes;

   /* from ratio */
   h_scale = 2.5;
   w_scale = 1.8789;

  // Get image from provider.
  if (kTfLiteOk != GetImage(H, W, C, tinyYoloV2::input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != tinyYoloV2::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }

  TfLiteTensor* outp = tinyYoloV2::interpreter->output(0);
 
  // Initialize output tensor with values
  float *out_data = outp->data.f;

  int grid_x, grid_y;
  float x, y, w, h;
  int x_min, x_max, y_min, y_max;
  for (int i = 0; i < 8 * 8; i++) {
      float conf = out_data[i * 6 + 4];
      float pred = sigmod(conf);
      // nms
      if (pred < 0.25)
          continue;
      x = (float)out_data[i * 6];
      y = (float)out_data[i * 6 + 1];
      w = (float)out_data[i * 6 + 2];
      h = (float)out_data[i * 6 + 3];
      grid_x = i % 8;
      grid_y = i / 8;
      x = (sigmod(x) + grid_x) * (256 / 8);
      y = (sigmod(y) + grid_y) * (256 / 8);
      w = expf(w) * anchors[0] * 256;
      h = expf(h) * anchors[1] * 256;

      x_min = (int)(x - w / 2);
      x_max = (int)(x + w / 2);
      y_min = (int)(y - h / 2);
      y_max = (int)(y + h / 2);


      boxes[num_boxes].x_min = x_min;
      boxes[num_boxes].x_max = x_max;
      boxes[num_boxes].y_min = y_min;
      boxes[num_boxes].y_max = y_max;
      boxes[num_boxes].score = pred;
      num_boxes++;

      printf("x_min: %d, y_min: %d, x_max: %d, y_max: %d, prediction = %0.2f\n",
        x_min, y_min, x_max, y_max, pred);

  }
  // 调用NMS
  nms(boxes, num_boxes, 0.4f, &out_boxes, &out_num_boxes);
  // 输出结果
  printf("After NMS, %d boxes are kept.\n", out_num_boxes);
  for (int i = 0; i < out_num_boxes; ++i) {
      int x_min = (int)(out_boxes[i].x_min * w_scale); 
      int y_min = (int)(out_boxes[i].y_min * h_scale);
      int x_max = (int)(out_boxes[i].x_max * w_scale);
      int y_max = (int)(out_boxes[i].y_max * h_scale);
      float conf = out_boxes[i].score;
      printf("Box %d: (%d, %d, %d, %d), Score: %.2f\n", i,
              x_min, 
              y_min, 
              x_max, 
              y_max, 
              conf);
  }
  // 释放内存
  free(out_boxes);
  return kTfLiteOk;
}
