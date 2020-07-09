/*
 * Copyright 2019 GreenWaves Technologies, SAS
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>

#ifdef RGB
	#include "mobv2_035_rgbKernels.h"
#else
	#include "mobv2_1_bwKernels.h"
#endif

#include "gaplib/ImgIO.h"

#define IMAGE_SIZE 		(CAMERA_WIDTH*CAMERA_HEIGHT*CAMERA_COLORS)
#define AT_INPUT_SIZE 	(AT_INPUT_WIDTH*AT_INPUT_HEIGHT*AT_INPUT_COLORS)

// Softmax always outputs Q15 short int even from 8 bit input
L2_MEM short int *ResOut;
AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash) = 0;
char *ImageName = NULL;
//u_int8_t *Input_1[AT_INPUT_SIZE];

static void RunNetwork()
{
  printf("Running on cluster\n");
#ifdef PERF
  printf("Start timer\n");
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif
  __PREFIX(CNN)(ResOut);
}


int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        printf("Usage: mnist [image_file]\n");
        exit(-1);
    }
    ImageName = argv[1];
/*-------------------------ALLOCATE THE OUTPUT TENSOR------------------------*/
	ResOut = (short int *) AT_L2_ALLOC(0, 2*sizeof(short int));
	if (ResOut==0) {
		printf("Failed to allocate Memory for Result (%ld bytes)\n", 2*sizeof(short int));
		return 1;
	}

/*--------------------CONSTRUCT THE NETWORK-------------------------*/
    printf("Constructor\n");
	// IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
	if (__PREFIX(CNN_Construct)())
	{
	  printf("Graph constructor exited with an error\n");
	  return 1;
	}
	printf("Constructor was OK!\n");

/* ----------------------------------------------------------- MAIN LOOP ---------------------------------------------------------------- */
	int count = 0;
	/*------------------- reading input data -----------------------------*/
	printf("Reading image from %s\n",ImageName);
	//Reading Image from Bridge
	#ifdef RGB
  		img_io_out_t type = IMGIO_OUTPUT_RGB565;
	#else
  		img_io_out_t type = IMGIO_OUTPUT_CHAR;  		
	#endif
	if (ReadImageFromFile(ImageName, AT_INPUT_WIDTH, AT_INPUT_HEIGHT, AT_INPUT_COLORS, Input_1, AT_INPUT_SIZE*sizeof(unsigned char), type, 0)) {
		printf("Failed to load image %s\n", ImageName);
		return 1;
	}
	printf("Finished reading image %s\n", ImageName);

	/*-----------------------CALL THE MAIN FUNCTION----------------------*/
	RunNetwork(NULL);
	if (ResOut[1] > ResOut[0]) {
	  printf("There is a vehicle! [whith score %d vs %d]\n", ResOut[1], ResOut[0]);
	} else {
	  printf("NO vehicle! [whith score %d vs %d]\n", ResOut[0], ResOut[1]);
	}

	/*-----------------------Desctruct the AT model----------------------*/
	__PREFIX(CNN_Destruct)();

	printf("Ended\n");
	return 0;
}
