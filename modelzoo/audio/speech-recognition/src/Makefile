TARGET = wav2letter_lstm

AI_MODELS_ROOT := ../../../../

# set REBUILD_TFLM = 0 means use tflm static library, now just support n300fd and nx900fd.a
# set REBUILD_TFLM = 1 means compile tflm by source which will cost a lot of time
REBUILD_TFLM ?= 0

CORE ?= n300fd

ARCH_EXT ?= 

DOWNLOAD ?= ddr

SRCDIRS = .

INCDIRS = .

NMSIS_LIB := nmsis_nn

STDCLIB ?= newlib_small

COMMON_FLAGS := -O3

include $(AI_MODELS_ROOT)/modelzoo/Makefile.common
