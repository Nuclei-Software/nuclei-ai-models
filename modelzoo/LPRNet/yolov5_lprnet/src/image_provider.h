/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef _IMAGE_PROVIDER_H_
#define _IMAGE_PROVIDER_H_

#include "tensorflow/lite/c/common.h"

#ifdef __cplusplus
extern "C" {
#endif

TfLiteStatus GetImage(int height, int weight, int channel, uint8_t* input_data);
TfLiteStatus GetImage_ocr(int xx, int yy, int ww, int hh, int channels, uint8_t* input_data);

#ifdef __cplusplus
}
#endif

#endif  // _IMAGE_PROVIDER_H_
