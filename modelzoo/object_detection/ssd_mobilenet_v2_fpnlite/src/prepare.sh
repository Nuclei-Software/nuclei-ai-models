#!/bin/bash

model_file="../pretrained_models/coco_2017_person/ssd_mobilenet_v2_fpnlite_035_224/ssd_mobilenet_v2_fpnlite_035_224_int8.tflite"
output_header="$(basename "$model_file" | tr './-' '_' | sed 's/\..*//')"


xxd -i "$model_file" > "${output_header}".h

echo "Generated model header: ${output_header}.h from $model_file"


python3 "../script/im2array.py"

echo "Generated test data OK"
