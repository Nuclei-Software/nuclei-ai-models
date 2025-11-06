#include <stdint.h>
#include <cstring>
#include "image_provider.h"
#include <cstdio>

extern signed char imgArray[196608];

TfLiteStatus GetImage(int height, int weight, int channel, uint8_t* input_data) {

  memcpy(input_data, imgArray, sizeof(imgArray));

  return kTfLiteOk;
}
