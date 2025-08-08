#!/usr/bin/env python3

"""This tool analyse an html visualization of a TensorFlow Lite and generate op_resolver code

then you can copy this code to your project

Example usage:

python generate_op_resolver_code xxx.tflite

See:
tflite-micro/tensorflow/lite/micro/kernels/micro_ops.h
tflite-micro/tensorflow/lite/micro/micro_mutable_op_resolver.h

"""

import re
import sys
from bs4 import BeautifulSoup
import visualize

micro_mutable_op_resolver_map = {
    "ABS":"Abs",
    "ADD":"Add",
    "ADD_N":"AddN",
    "ARG_MAX":"ArgMax",
    "ARG_MIN":"ArgMin",
    "ASSIGN_VARIABLE":"AssignVariable",
    "AVERAGE_POOL_2D":"AveragePool2D",
    "BATCH_MATMUL":"BatchMatMul",
    "BATCH_TO_SPACE_ND":"BatchToSpaceNd",
    "BROADCAST_ARGS":"BroadcastArgs",
    "BROADCAST_TO":"BroadcastTo",
    "CALL_ONCE":"CallOnce",
    "CAST":"Cast",
    "CEIL":"Ceil",
    "CIRCULAR_BUFFER":"CircularBuffer",
    "CONCATENATION":"Concatenation",
    "CONV_2D":"Conv2D",
    "COS":"Cos",
    "CUMSUM":"CumSum",
    "DEPTH_TO_SPACE":"DepthToSpace",
    "DEPTHWISE_CONV_2D":"DepthwiseConv2D",
    "DEQUANTIZE":"Dequantize",
    "DIV":"Div",
    "ELU":"Elu",
    "EMBEDDING_LOOKUP":"EmbeddingLookup",
    "EQUAL":"Equal",
    "ETHOSU":"EthosU",
    "EXP":"Exp",
    "EXPAND_DIMS":"ExpandDims",
    "FILL":"Fill",
    "FLOOR":"Floor",
    "FLOOR_DIV":"FloorDiv",
    "FLOOR_MOD":"FloorMod",
    "FULLY_CONNECTED":"FullyConnected",
    "GATHER":"Gather",
    "GATHER_ND":"GatherNd",
    "GREATER":"Greater",
    "GREATER_EQUAL":"GreaterEqual",
    "HARD_SWISH":"HardSwish",
    "IF":"If",
    "L2_NORMALIZATION":"L2Normalization",
    "L2_POOL_2D":"L2Pool2D",
    "LEAKY_RELU":"LeakyRelu",
    "LESS":"Less",
    "LESS_EQUAL":"LessEqual",
    "LOG":"Log",
    "LOG_SOFTMAX":"LogSoftmax",
    "LOGICAL_AND":"LogicalAnd",
    "LOGICAL_NOT":"LogicalNot",
    "LOGICAL_OR":"LogicalOr",
    "LOGISTIC":"Logistic",
    "MAX_POOL_2D":"MaxPool2D",
    "MAXIMUM":"Maximum",
    "MEAN":"Mean",
    "MINIMUM":"Minimum",
    "MIRROR_PAD":"MirrorPad",
    "MUL":"Mul",
    "NEG":"Neg",
    "NOT_EQUAL":"NotEqual",
    "PACK":"Pack",
    "PAD":"Pad",
    "PADV2":"PadV2",
    "PRELU":"Prelu",
    "QUANTIZE":"Quantize",
    "READ_VARIABLE":"ReadVariable",
    "REDUCE_MAX":"ReduceMax",
    "RELU":"Relu",
    "RELU6":"Relu6",
    "RESHAPE":"Reshape",
    "RESIZE_BILINEAR":"ResizeBilinear",
    "RESIZE_NEAREST_NEIGHBOR":"ResizeNearestNeighbor",
    "SignalRfft":"Rfft",
    "ROUND":"Round",
    "RSQRT":"Rsqrt",
    "SELECT_V2":"SelectV2",
    "SHAPE":"Shape",
    "SIN":"Sin",
    "SLICE":"Slice",
    "SOFTMAX":"Softmax",
    "SPACE_TO_BATCH_ND":"SpaceToBatchNd",
    "SPACE_TO_DEPTH":"SpaceToDepth",
    "SPLIT":"Split",
    "SPLIT_V":"SplitV",
    "SQRT":"Sqrt",
    "SQUARE":"Square",
    "SQUARED_DIFFERENCE":"SquaredDifference",
    "SQUEEZE":"Squeeze",
    "STRIDED_SLICE":"StridedSlice",
    "SUB":"Sub",
    "SUM":"Sum",
    "SVDF":"Svdf",
    "TANH":"Tanh",
    "TRANSPOSE":"Transpose",
    "TRANSPOSE_CONV":"TransposeConv",
    "UNIDIRECTIONAL_SEQUENCE_LSTM":"UnidirectionalSequenceLSTM",
    "UNPACK":"Unpack",
    "VAR_HANDLE":"VarHandle",
    "WHILE":"While",
    "ZEROS_LIKE":"ZerosLike",

    # tflm_signal
    "SignalDelay":"Delay",
    "SignalFftAutoScale":"Delay",
    "SignalFilterBank":"FilterBank",
    "SignalFilterBankLog":"FilterBankLog",
    "SignalFilterBankSpectralSubtraction":"FilterBankSpectralSubtraction",
    "SignalFilterBankSquareRoot":"FilterBankSquareRoot",
    "SignalEnergy":"Energy",
    "SignalFramer":"Framer",
    "SignalOverlapAdd":"OverlapAdd",
    "SignalPCAN":"PCAN",
    "SignalStacker":"Stacker",
    "SignalWindow":"Window",
    "SignalIrfft":"Irfft",
}

def analyse_html(html):
    buffer_total = 0
    Operator_table = []

    html_content = str(BeautifulSoup(html, 'html.parser'))
    switch1, switch2 = False, False
    for line in html_content.splitlines():
        buffer_flag = line.startswith('</div><h2>Buffers</h2>')
        opcode_flag = line.startswith('<h2>Operator Codes</h2>')
        end_flag = line.startswith('</body></html>')
        #print(line)

        if buffer_flag and not opcode_flag:
            switch1, switch2 = True, False
        if opcode_flag:
            switch1, switch2 = False, True
        if end_flag:
            switch1, switch2 = False, False
        if switch1:
            buffer_record = re.search(r'(\d+)\s*bytes', line)
            if buffer_record:
                buffer_total += int(buffer_record.group(1))
        if switch2:
            pattern_builtin = re.compile(r'<td>\d+</td><td>(.*?)</td>')
            pattern_custom = re.compile(r'<td>(Signal\w+)</td>')
            # builtin_code
            matches = re.search(pattern_builtin, line)
            if matches and matches.group(1) != 'CUSTOM':
                Operator_table.append(matches.group(1))
            else:
            # custom_code
                matches = re.search(pattern_custom, line)
                if matches:
                    Operator_table.append(matches.group(1))

    print(f'Buffers Total Need {buffer_total} bytes')

    return Operator_table

def generate_op_resolver_code(Operator_table):
    print(f'Operator number is {len(Operator_table)}')
    for op in Operator_table:
        code = ""
        if op not in micro_mutable_op_resolver_map.keys():
            print(f'Unsupport {op} !!')
            #break
        else:
            code = 'if (op_resolver.Add' + micro_mutable_op_resolver_map[op] + '() != kTfLiteOk) {' + '\n'
            code += '    ' + 'printf("Add ' + micro_mutable_op_resolver_map[op] + ' fail!\\n");' + '\n'
            code += '    ' + 'return;' + '\n'
            code += '}'
            print(code)
    

def main(argv):
    tflite_input = argv[1]
    if not tflite_input.endswith('.tflite'):
        print("Usage: %s <xxx.tflite>" % (argv[0]))
        return

    html = visualize.create_html(tflite_input)
    Operator_table = analyse_html(html)
    generate_op_resolver_code(Operator_table)

if __name__ == "__main__":
  main(sys.argv)
