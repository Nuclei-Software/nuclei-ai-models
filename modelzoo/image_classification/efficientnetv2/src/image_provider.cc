#include <stdint.h>
#include <cstring>
#include "image_provider.h"
#include <cstdio>

extern unsigned char imgArray[150528];

TfLiteStatus GetImage(int height, int weight, int channel, uint8_t* input_data) {

  memcpy(input_data, imgArray, sizeof(imgArray));

  return kTfLiteOk;
}