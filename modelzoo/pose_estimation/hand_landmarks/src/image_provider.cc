#include <stdint.h>
#include <cstring>
#include "image_provider.h"
#include <cstdio>

extern uint8_t imgArray[150528];

TfLiteStatus GetImage(int height, int weight, int channel, uint8_t* input_data) {
  int img_size = height * weight * channel;
  memcpy(input_data, imgArray, img_size);
  return kTfLiteOk;
}
