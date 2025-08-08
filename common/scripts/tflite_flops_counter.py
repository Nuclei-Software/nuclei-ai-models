#!/usr/bin/env python3

import sys
import numpy as np
import tensorflow.lite as tflite

def calculate_flops(op_name, input_shape, output_shape, filter_shape=None):
    """
    Calculate FLOPs based on the operator name and shape
    """
    if op_name == "CONV_2D":
        if filter_shape is None:
            return 0
        _, kh, kw, ic = filter_shape    # filter: (oc, kh, kw, ic)
        _, oh, ow, oc = output_shape    # output: (b, oh, ow, oc)
        macs_per_output = kh * kw * ic
        total_flops = 2 * oh * ow * oc * macs_per_output
        return int(total_flops)

    elif op_name == "DEPTHWISE_CONV_2D":
        if filter_shape is None:
            return 0
        _, kh, kw, ic = filter_shape    # depthwise: (1, kh, kw, ic)
        _, oh, ow, oc = output_shape    # output: (b, oh, ow, oc)
        macs_per_output = kh * kw * 1
        total_flops = 2 * oh * ow * oc * macs_per_output
        return int(total_flops)

    elif op_name == "FULLY_CONNECTED":
        ic = input_shape[-1]
        oc = output_shape[-1]
        flops = 2 * ic * oc
        return int(flops)

    elif op_name == "ADD" or op_name == "SUB" or op_name == "MUL" or op_name == "DIV":
        # element add/mul/sub/div
        flops = np.prod(output_shape)
        return int(flops)

    elif op_name == "AVERAGE_POOL_2D" or op_name == "MAX_POOL_2D":
        _, oh, ow, oc = output_shape    # output: (b, oh, ow, oc)
        flops = oh * ow * oc            # roughly press 1 FLOP per output
        return int(flops)

    else:
        # Ignore 
        return 0

def get_tensor_shape(interpreter, tensor_idx):
    if tensor_idx == -1:
        return None
    tensor = interpreter.get_tensor_details()[tensor_idx]
    return tensor['shape']

def get_filter_shape(interpreter, weight_tensor_idx):
    if weight_tensor_idx == -1:
        return None
    weight = interpreter.get_tensor(weight_tensor_idx)
    return weight.shape

def analyze_model_flops(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    ops = interpreter._get_ops_details()
    tensor_details = interpreter.get_tensor_details()

    total_flops = 0
    op_flops_list = []

    for op in ops:
        op_name = op['op_name']
        inputs = op['inputs']
        outputs = op['outputs']

        input_shape = get_tensor_shape(interpreter, inputs[0]) if len(inputs) > 0 else None
        output_shape = get_tensor_shape(interpreter, outputs[0]) if len(outputs) > 0 else None

        filter_shape = None
        if op_name in ["CONV_2D", "DEPTHWISE_CONV_2D"]:
            if len(inputs) >= 2:
                filter_shape = get_filter_shape(interpreter, inputs[1])

        flops = calculate_flops(
            op_name=op_name,
            input_shape=input_shape,
            output_shape=output_shape,
            filter_shape=filter_shape,
        )

        total_flops += flops
        op_flops_list.append({
            'op': op_name,
            'input_shape': input_shape,
            'output_shape': output_shape,
            'flops': flops
        })

        print(f"|{op_name:15} | In: {input_shape!s:25} | Out: {output_shape!s:25} | FLOPs: {flops:>12,} |")

    print("\n" + "="*80)
    print(f"TFlite: {model_path}")
    print(f"Total: {total_flops:,} FLOPs")
    print(f"Total: {total_flops / 1e6:.2f} MFLOPs")
    print(f"Total: {total_flops / 1e9:.2f} GFLOPs")
    print("="*80)

    return total_flops

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python tflite_flops_counter.py <model.tflite>")
        sys.exit(1)

    model_path = sys.argv[1]
    analyze_model_flops(model_path)   
