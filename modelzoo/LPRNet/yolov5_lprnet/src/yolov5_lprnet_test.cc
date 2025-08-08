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
#include "yolov5_best_int8.h"
#include "image_provider.h"


namespace yolo {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 5 * 1024 * 1024;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup_yolov5 () {
  tflite::InitializeTarget();
  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  yolo::model = tflite::GetModel(___pretrained_models_yolov5_best_int8_tflite);
  if (yolo::model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf(
        "Model provided is schema version %d not equal "
        "to supported version %d.\n",
        yolo::model->version(), TFLITE_SCHEMA_VERSION);
  }

  static tflite::MicroMutableOpResolver<11>  op_resolver;
  if (op_resolver.AddQuantize() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddPad() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddConv2D() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddLogistic() != kTfLiteOk) {
    return;
  }

  if (op_resolver.AddMul() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddAdd() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddConcatenation() != kTfLiteOk) {
    return;
  }
    if (op_resolver.AddMaxPool2D() != kTfLiteOk) {
    return;
  }

  if (op_resolver.AddResizeNearestNeighbor() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddReshape() != kTfLiteOk) {
    return;
  }
  if (op_resolver.AddStridedSlice() != kTfLiteOk) {
    return;
  }

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      yolo::model, op_resolver, yolo::tensor_arena, yolo::kTensorArenaSize);
  yolo::interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = yolo::interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }

  // Obtain a pointer to the model's input tensor
  yolo::input = yolo::interpreter->input(0);

  // Make sure the input has the properties we expect
  if (yolo::input == nullptr) {
    MicroPrintf("Input tensor is null.");
    return;
  }
}

namespace lprnet {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  // An area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 1266753;
  static uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// The name of this function is important for Arduino compatibility.
void setup_lprnet() {
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
    // 首先根据置信度分数进行降序排序
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


const char chars[84][6] = {"jing", "hu", "jin", "yu", "ji", "ji", "meng", "liao", "ji", "hei", "su", "zhe", "wan", "min", "gan", "lu", "yu", "e", "xiang", "yue", "gui",
         "qiong", "chuan", "gui", "yun", "zang", "shan", "gan", "qing", "ning", "xin", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "A",
         "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X",
         "Y", "Z","gang","xue","shi","jing","ao","gua","jun","bei","nan","guang","shen","lan","cheng","ji","hai","ming","hang","kong"
};

int main(int argc, char* argv[]) {
  setup_yolov5();
  int H, W, C;

  H = 640;
  W = 640;
  C = 3;

  // Get image from provider.
  if (kTfLiteOk != GetImage(W, H, C, yolo::input->data.uint8)) {
    MicroPrintf("Image capture failed.");
  }

  // Run the model on this input and make sure it succeeds.
  if (kTfLiteOk != yolo::interpreter->Invoke()) {
    MicroPrintf("Invoke failed.");
  }
  MicroPrintf("Invoke complete.");
  TfLiteTensor* outp = yolo::interpreter->output(0);
  uint8_t *out_data = outp->data.uint8;

  float x, y, w, h;
  int x_min, x_max, y_min, y_max;
  // float padh = 0.125, padw = 0;
  float max_shape = 640;
  int num_boxes = 0;
  Bbox* out_boxes;
  int out_num_boxes;

  #define YOLO_SCALE 0.0074273087084293365
  #define ZERO_POINT -2

  for (int i = 0; i < 25200 * 6;  i+= 6) {
    float conf = (out_data[i + 4]+ ZERO_POINT) * YOLO_SCALE;
    float class_scores =  (out_data[i + 4]+ ZERO_POINT) * YOLO_SCALE;
    if (conf * class_scores < 0.25)
      continue;
    x = (out_data[i] + ZERO_POINT) * YOLO_SCALE;
    y = (out_data[i + 1] + ZERO_POINT) * YOLO_SCALE;
    w = (out_data[i + 2] + ZERO_POINT) * YOLO_SCALE;
    h = (out_data[i + 3] + ZERO_POINT) * YOLO_SCALE;

    x_min = (x - w / 2) * max_shape;
    x_max = (x + w / 2) * max_shape;
    y_min = (y - h / 2) * max_shape;
    y_max = (y + h / 2) * max_shape;

    boxes[num_boxes].x_min = x_min;
    boxes[num_boxes].x_max = x_max;
    boxes[num_boxes].y_min = y_min;
    boxes[num_boxes].y_max = y_max;
    boxes[num_boxes].score = conf;
    num_boxes++;
    

    // printf("x_min: %d, y_min: %d, w: %d, h: %d, prediction = %0.2f\n",
    //     x_min, y_min, x_max - x_min, y_max - y_min, conf);
    
  }

  // 调用NMS
  nms(boxes, num_boxes, 0.45f, &out_boxes, &out_num_boxes);

  setup_lprnet();
  // 输出结果
  printf("After NMS, %d boxes are kept.\n", out_num_boxes);
  for (int i = 0; i < out_num_boxes; ++i) {
      printf("Box %d: (%d, %d, %d, %d), Score: %.2f\n", i,
      out_boxes[i].x_min, out_boxes[i].y_min, out_boxes[i].x_max, out_boxes[i].y_max, out_boxes[i].score);

      x = out_boxes[i].x_min;
      y = out_boxes[i].y_min;
      w = out_boxes[i].x_max - out_boxes[i].x_min;
      h = out_boxes[i].y_max - out_boxes[i].y_min;
      // Get image from provider.
      if (kTfLiteOk != GetImage_ocr(x, y, w, h, C, lprnet::input->data.uint8)) {
        MicroPrintf("ocr Image capture failed.");
      }
    
      // Run the model on this input and make sure it succeeds.
      if (kTfLiteOk != lprnet::interpreter->Invoke()) {
        MicroPrintf("Invoke failed.");
      }
    
      outp = lprnet::interpreter->output(0);
      
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
  }
  // 释放内存
  free(out_boxes);

  return kTfLiteOk;

}
