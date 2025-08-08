#include <stdint.h>
#include <cstring>
#include "image_provider.h"
#include <cstdio>
#include <math.h>

extern unsigned char imgArray[640 * 640 * 3];

TfLiteStatus GetImage(int height, int weight, int channel, uint8_t* input_data) {

  memcpy(input_data, imgArray, sizeof(imgArray));

  return kTfLiteOk;
}

#if 0
TfLiteStatus GetImage_ocr(int xx, int yy, int ww, int hh, int channels, uint8_t* input_data) {

  int32_t index = 0;

  int h = 40;
  int w = 160;
  int ih = hh;
  int iw = ww;

  float r = fmin(float(h) / ih, float(w) / iw);
  int nw = w;
  int nh = h;
  int pad_w = (w - nw) / 2;
  int pad_h = (h - nh) / 2;
  
  printf("%d %d %d %d %d %d\r\n", xx, yy, ww, hh, nw, nh);
  memset(input_data, 255, w * h * channels);
  for (int x = pad_w; x < nw; x++) {
    for (int y = pad_h; y < nh; y++) {
      for (int c = 0; c < 3; c++) {
        input_data[index] = imgArray[(yy + (int)(y / r)) * 640 * 3 + (xx + (int)(x / r)) * 3 + c];
        index++;
      }
    }
  }

  return kTfLiteOk;
}
#else
TfLiteStatus GetImage_ocr(int xx, int yy, int ww, int hh, int channels, uint8_t* input_data) {

  int32_t index = 0;

  int h = 40;
  int w = 160;

  float rh = float(hh) / h;
  float rw = float(ww) / w;
  
  printf("(%d %d %d %d) => (%d %d)\r\n", xx, yy, ww, hh, h, w);
  memset(input_data, 255, h * w *  channels);
  for (int x = 0; x < w; x++) {
    for (int y = 0; y < h; y++) {
      for (int c = 0; c < 3; c++) {
        int src_x = xx + int(x * rw);
        int src_y = yy + int(y * rh);
        input_data[index] = imgArray[src_y * 640 * 3 + src_x * 3 + c];
        index++;
      }
    }
  }

  return kTfLiteOk;
}
#endif
