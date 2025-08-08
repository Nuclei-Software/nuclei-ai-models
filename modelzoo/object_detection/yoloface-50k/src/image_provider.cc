#include <stdint.h>
#include <cstring>
#include "image_provider.h"
#include <cstdio>

extern int8_t imgArray[9408];

TfLiteStatus GetImage(int height, int weight, int channel, int8_t* input_data) {

  memcpy(input_data, imgArray, sizeof(imgArray));

  return kTfLiteOk;
}
