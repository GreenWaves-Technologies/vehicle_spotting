# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

MODEL_SUFFIX?=
MODEL_PREFIX?=

# The training of the model is slightly different depending on
# the quantization. This is because in 8 bit mode we used signed
# 8 bit so the input to the model needs to be shifted 1 bit
ifeq ($(QUANT_BITS),8)
  PROJ_FOLDER=GAP_PORTING_8BIT
else
  ifeq ($(QUANT_BITS),16)
    PROJ_FOLDER=GAP_PORTING_16BIT
  else
    $(error Dont know how to build with this bit width)
  endif
endif


MODEL_PYTHON=python

# Increase this to improve accuracy
TRAINING_EPOCHS?=1
MODEL_COMMON ?= ./utils
MODEL_COMMON_INC ?= $(MODEL_COMMON)/
MODEL_COMMON_SRC ?= $(MODEL_COMMON)/
MODEL_COMMON_SRC_FILES ?= ImgIO.c helpers.c
MODEL_COMMON_SRCS = $(realpath $(addprefix $(MODEL_COMMON_SRC)/,$(MODEL_COMMON_SRC_FILES)))

MODEL_TFLITE = $(PROJ_FOLDER)/nntool/$(MODEL_PREFIX).tflite

TENSORS_DIR = $(PROJ_FOLDER)/tensors
MODEL_TENSORS = $(PROJ_FOLDER)/$(MODEL_PREFIX)_L3_Flash_Const.dat

MODEL_STATE = $(PROJ_FOLDER)/nntool/$(MODEL_PREFIX).json
ifeq ($(PERF),1)
	MODEL_SRC = $(MODEL_PREFIX)Model_perf.c
else
	MODEL_SRC = $(MODEL_PREFIX)Model.c
endif
MODEL_GEN = $(PROJ_FOLDER)/$(MODEL_PREFIX)Kernels 
MODEL_GEN_C = $(addsuffix .c, $(MODEL_GEN))
MODEL_GEN_CLEAN = $(MODEL_GEN_C) $(addsuffix .h, $(MODEL_GEN))
MODEL_GEN_EXE = $(PROJ_FOLDER)/GenTile

MODEL_GENFLAGS_EXTRA =

EXTRA_GENERATOR_SRC =

# NN Tool Configuration
NNTOOL=nntool
NNTOOL_VALIDATION ?= 0
$(info script $(NNTOOL_SCRIPT))
ifndef NNTOOL_SCRIPT
  ifeq ($(NNTOOL_VALIDATION), 1)
    NNTOOL_SCRIPT=$(PROJ_FOLDER)/nntool/nntool_script_w_validation
  else
    NNTOOL_SCRIPT=$(PROJ_FOLDER)/nntool/nntool_script
  endif
endif

#others
RM=rm -f


# Autotiler Configuration
MODEL_SRCS += $(MODEL_GEN_C)
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_BiasReLULinear_BasicKernels.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_Conv_BasicKernels.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_Conv_DP_BasicKernels.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_Conv_DW_BasicKernels.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_Conv_DW_DP_BasicKernels.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_Pooling_BasicKernels.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_MatAlgebra.c
MODEL_SRCS += $(TILER_CNN_KERNEL_PATH)/CNN_SoftMax.c

MODEL_SIZE_CFLAGS = -DAT_INPUT_HEIGHT=$(AT_INPUT_HEIGHT) -DAT_INPUT_WIDTH=$(AT_INPUT_WIDTH) -DAT_INPUT_COLORS=$(AT_INPUT_COLORS)
