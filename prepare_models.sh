#!/bin/env bash

if ! xxd --version ; then
    echo -e "\033[1;31mERROR: xxd is not installed, check https://linux.die.net/man/1/xxd and install it\033[0m"
    exit 1
fi

echo "Generate model src code from tflite models using xxd tool"
set -x

xxd -i modelzoo/audio/speech-recognition/pretrained_models/wav2letter/wav2letter_int8.tflite  modelzoo/audio/speech-recognition/src/wav2letter_int8.cc
xxd -i modelzoo/image_classification/efficientnetv2/pretrained_models/ImageNet/efficientnet_v2B0_224_int8.tflite  modelzoo/image_classification/efficientnetv2/src/efficientnet_v2B0_224_int8.cc
xxd -i modelzoo/image_classification/resnet50v2/pretrained_models/ImageNet/resnet50_v2_224/resnet50_v2_224_int8.tflite  modelzoo/image_classification/resnet50v2/src/resnet50_v2_224_int8.cc
xxd -i modelzoo/object_detection/tiny_yolo_v2/pretrained_models/coco_2017_person/tiny_yolo_v2_224_int8.tflite  modelzoo/object_detection/tiny_yolo_v2/src/tiny_yolo_v2_224_int8.cc

set +x
echo -e "\033[1;32mINFO: Required source files are generated now\033[0m"
