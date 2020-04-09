# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

ifndef GAP_SDK_HOME
  $(error Source sourceme in gap_sdk first)
endif

pulpChip=GAP
MODEL_PREFIX=mbv1_grayscale
APP = vww_vehicle

PERF ?= 0
QUANT_BITS?=8
ALREADY_FLASHED?=0
IMAGE?=$(CURDIR)/himax_1.ppm

export GAP_USE_OPENOCD=1
io=host

include ./model_decl.mk

# Here we set the memory allocation for the generated kernels
# REMEMBER THAT THE L1 MEMORY ALLOCATION MUST INCLUDE SPACE
# FOR ALLOCATED STACKS!
CLUSTER_STACK_SIZE?=2048
CLUSTER_SLAVE_STACK_SIZE?=512

TOTAL_STACK_SIZE=$(shell expr $(CLUSTER_STACK_SIZE) \+ $(CLUSTER_SLAVE_STACK_SIZE) \* 7)
MODEL_L1_MEMORY=$(shell expr 60000 \- $(TOTAL_STACK_SIZE))
MODEL_L2_MEMORY?=500000
MODEL_L3_MEMORY?=8388608
# hram - HyperBus RAM
# qspiram - Quad SPI RAM
MODEL_L3_EXEC=hram
# hflash - HyperBus Flash
# qpsiflash - Quad SPI Flash
MODEL_L3_CONST=hflash


READFS_FILES += $(realpath $(MODEL_TENSORS))

#This is to avoid flaching every time model data into flash
ifneq ($(ALREADY_FLASHED),1)	
	PLPBRIDGE_FLAGS += -f 
endif


APP_SRCS += $(PROJ_FOLDER)/main.c $(MODEL_COMMON_SRCS) $(MODEL_SRCS)

APP_CFLAGS += -O3 -s -mno-memcpy -fno-tree-loop-distribute-patterns 
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) -I$(TILER_CNN_KERNEL_PATH) -I$(PROJ_FOLDER)/.
APP_CFLAGS += -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)
APP_CFLAGS += -DAT_IMAGE=$(IMAGE) -DAT_IN_HEIGHT=224 -DAT_IN_WIDTH=224 -DAT_IN_COLORS=1
ifeq ($(PERF), 1)
	APP_CFLAGS += -DPERF
endif

# all depends on the model
all:: model

clean:: clean_model

include ./model_rules.mk
include $(RULES_DIR)/pmsis_rules.mk

