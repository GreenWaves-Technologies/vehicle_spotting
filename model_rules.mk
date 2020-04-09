# Copyright (C) 2017 GreenWaves Technologies
# All rights reserved.

# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

USE_DISP=1
ifdef USE_DISP
  SDL_FLAGS= -lSDL2 -lSDL2_ttf
else
  SDL_FLAGS=
endif

ifdef MODEL_L1_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L1 $(MODEL_L1_MEMORY)
endif

ifdef MODEL_L2_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L2 $(MODEL_L2_MEMORY)
endif

ifdef MODEL_L3_MEMORY
  MODEL_GEN_EXTRA_FLAGS += --L3 $(MODEL_L3_MEMORY)
endif

# Creates an NNTOOL state file by running the commands in the script
# These commands could be run interactively
# The commands:
# 	Adjust the model to match AutoTiler tensor order
#	Fuse nodes together to match fused AutoTiler generators
#	Auto quantify the graph
#	Save the graph state files

$(MODEL_STATE): 
	echo "GENERATING NNTOOL STATE FILE"
	$(NNTOOL) -s $(NNTOOL_SCRIPT) $(MODEL_TFLITE)

nntool_state: $(MODEL_STATE)

# Runs NNTOOL with its state file to generate the autotiler model code
$(PROJ_FOLDER)/$(MODEL_SRC): $(MODEL_STATE) clean_tensors
	echo "GENERATING AUTOTILER MODEL"
	$(NNTOOL) -g -M $(PROJ_FOLDER) -m $(MODEL_SRC) -T $(TENSORS_DIR) $(MODEL_STATE)

nntool_gen: $(MODEL_SRC)

# Build the code generator from the model code
$(MODEL_GEN_EXE): $(PROJ_FOLDER)/$(MODEL_SRC)
	echo "COMPILING AUTOTILER MODEL"
	gcc -g -o $(MODEL_GEN_EXE) -I. -I$(TILER_INC) -I$(TILER_EMU_INC) -I$(TILER_CNN_GENERATOR_PATH) $(PROJ_FOLDER)/$(MODEL_SRC) $(TILER_CNN_GENERATOR_PATH)/CNN_Generators.c $(EXTRA_GENERATOR_SRC) $(TILER_LIB) $(SDL_FLAGS)

compile_model: $(MODEL_GEN_EXE)

# Run the code generator to generate GAP graph and kernel code
$(MODEL_GEN_C): $(MODEL_GEN_EXE)
	echo "RUNNING AUTOTILER MODEL"
	$(MODEL_GEN_EXE) -o $(PROJ_FOLDER) -c $(PROJ_FOLDER) #$(MODEL_GEN_EXTRA_FLAGS)

# A phony target to simplify including this in the main Makefile
model: $(MODEL_GEN_C)

clean_tensors:
	$(RM) -rf $(TENSORS_DIR)

clean_model:
	$(RM) $(PROJ_FOLDER)/$(MODEL_GEN_EXE)
	$(RM) $(MODEL_SRC)
	$(RM) $(PROJ_FOLDER)/*Kernels.c
	$(RM) $(PROJ_FOLDER)/*Kernels.h
	$(RM) $(PROJ_FOLDER)/*.dat

.PHONY: model clean_model clean_train test_images clean_images train nntool_gen nntool_state tflite compile_model
