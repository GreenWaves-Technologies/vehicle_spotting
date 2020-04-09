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
#include <limits.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "mbv1_grayscaleKernels.h"

//#include "setup.h"
#include "ImgIO.h"
//#include "cascade.h"
//#include "display.h"

#ifdef __EMUL__
 #ifdef PERF
  #undef PERF
 #endif
#else
 #include "pmsis.h"
 #include "bsp/flash/hyperflash.h"
 #include "bsp/bsp.h"
 #include "bsp/buffer.h"
 #include "bsp/camera/himax.h"
 #include "bsp/camera/mt9v034.h"
 #include "bsp/ram/hyperram.h"
#endif


#ifndef HAVE_CAMERA
 #define __XSTR(__s) __STR(__s)
 #define __STR(__s) #__s
 #define IMAGE_SIZE (CAMERA_WIDTH*CAMERA_HEIGHT*CAMERA_COLORS)
#endif

#define AT_IN_SIZE (AT_IN_WIDTH*AT_IN_HEIGHT*AT_IN_COLORS)

// Softmax always outputs Q15 short int even from 8 bit input
L2_MEM short int *ResOut;
L2_MEM short int imgin_unsigned_short[AT_IN_SIZE];
L2_MEM short int *imgin_signed_short = imgin_unsigned_short;

AT_HYPERFLASH_FS_EXT_ADDR_TYPE mbv1_grayscale_L3_Flash = 0;

#ifdef PERF
L2_MEM rt_perf_t *cluster_perf;
#endif

static void RunNetwork()
{
  printf("Running on cluster\n");
#ifdef PERF
  printf("Start timer\n");
  gap_cl_starttimer();
  gap_cl_resethwtimer();
#endif
  mbv1_grayscaleCNN(imgin_signed_short, ResOut);
  printf("Runner completed\n");
  printf("\n");

  //Checki Results
  if (ResOut[1] > ResOut[0]) {
    printf("vehicle seen (%d, %d)\n", ResOut[0], ResOut[1]);
  } else {
    printf("no vehicle seen (%d, %d)\n", ResOut[0], ResOut[1]);
  }
  printf("\n");
}

#if defined(__EMUL__)
int main(int argc, char *argv[]) 
{
  if (argc < 2) {
    printf("Usage: %s [image_file]\n", argv[0]);
    exit(1);
  }
  char *ImageName = argv[1];
#else
int body(void)
{
    static pi_buffer_t RenderBuffer;
    struct pi_device camera;
    struct pi_device display;	
    int display_orientation = PI_ILI_ORIENTATION_270;

/*------------------------HyperRAM---------------------------*/
	printf("Configuring Hyperram..\n");
	struct pi_device HyperRam;
	struct pi_hyperram_conf hyper_conf;

	pi_hyperram_conf_init(&hyper_conf);
	pi_open_from_conf(&HyperRam, &hyper_conf);

	if (pi_ram_open(&HyperRam))
	{
	  printf("Error: cannot open Hyperram!\n");
	  pmsis_exit(-2);
	}

	printf("HyperRAM config done\n");

	// The hyper chip need to wait a bit.
	// TODO: find out need to wait how many times.
	pi_time_wait_us(1*1000*1000);

/*----------------------HyperFlash & FS-----------------------*/
	printf("Configuring Hyperflash and FS..\n");
	struct pi_device fs;
	struct pi_device flash;
	struct pi_fs_conf fsconf;
	struct pi_hyperflash_conf flash_conf;
	pi_fs_conf_init(&fsconf);

	pi_hyperflash_conf_init(&flash_conf);
	pi_open_from_conf(&flash, &flash_conf);

	if (pi_flash_open(&flash))
	{
	  printf("Error: Flash open failed\n");
	  pmsis_exit(-3);
	}
	fsconf.flash = &flash;

	pi_open_from_conf(&fs, &fsconf);

	int error = pi_fs_mount(&fs);
	if (error)
	{
	  printf("Error: FS mount failed with error %d\n", error);
	  pmsis_exit(-3);
	}

	printf("FS mounted\n");

/*-----------------voltage-frequency settings-----------------------*/
	rt_freq_set(RT_FREQ_DOMAIN_FC, 50000000);
	rt_freq_set(RT_FREQ_DOMAIN_CL,150000000);
	PMU_set_voltage(1200,0);

/*-------------------OPEN THE CLUSTER-------------------------------*/
	struct pi_device cluster_dev;
	struct pi_cluster_conf conf;
	pi_cluster_conf_init(&conf);
	pi_open_from_conf(&cluster_dev, (void *)&conf);
	pi_cluster_open(&cluster_dev);

/*--------------------------TASK SETUP------------------------------*/
	struct pi_cluster_task *task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
	if(task==NULL) {
	  printf("pi_cluster_task alloc Error!\n");
	  pmsis_exit(-1);
	}
	memset(task, 0, sizeof(struct pi_cluster_task));
	task->entry = &RunNetwork;
	task->stack_size = STACK_SIZE;
	task->slave_stack_size = SLAVE_STACK_SIZE;
	task->arg = NULL;

#endif //Not Emulator
	
/*--------------------CONSTRUCT THE NETWORK-------------------------*/
    printf("Constructor\n");
	// IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
	if (mbv1_grayscaleCNN_Construct())
	{
	  printf("Graph constructor exited with an error\n");
	  return 1;
	}


/*-------------------reading input data-----------------------------*/
	#ifdef HAVE_CAMERA
		//OPEN HAVE_CAMERA 
        pi_camera_control(&camera, PI_CAMERA_CMD_START, 0);
        pi_camera_capture(&camera, imgin_unsigned_short, IMAGE_SIZE);
        pi_camera_control(&camera, PI_CAMERA_CMD_STOP, 0);
	#else
        #ifndef __EMUL__
  			char *ImageName = __XSTR(AT_IMAGE);
  		#endif
		printf("Reading image\n");
		//Reading Image from Bridge
		if (ReadImageFromFile(ImageName, AT_IN_WIDTH, AT_IN_HEIGHT, 1, imgin_unsigned_short, AT_IN_SIZE*sizeof(unsigned char), 0, 1)) {
			printf("Failed to load image %s\n", ImageName);
			return 1;
		}
		printf("Finished reading image %s\n", ImageName);
	#endif

/*--------------------convert to signed in [-128:127]----------------*/
	for(int i=0; i<AT_IN_SIZE; i++){
		imgin_signed_short[i] = ((short int) ( ((int) (imgin_unsigned_short[i])) - 128)) << 8;
	}

/*----------------------ALLOCATE THE OUTPUT TENSOR-------------------*/
	ResOut = (short int *) AT_L2_ALLOC(0, 2*sizeof(short int));
	if (ResOut==0) {
		printf("Failed to allocate Memory for Result (%ld bytes)\n", 2*sizeof(short int));
		return 1;
	}

/*-----------------------CALL THE MAIN FUNCTION----------------------*/
	printf("Call cluster\n");
#ifdef __EMUL__
	RunNetwork(NULL);
#else
	pi_cluster_send_task_to_cl(&cluster_dev, task);
#endif

/*-----------------------Desctruct the AT model----------------------*/
	mbv1_grayscaleCNN_Destruct();

#ifdef PERF
/*------------------------Performance Counter------------------------*/
	{
		unsigned int TotalCycles = 0, TotalOper = 0;
		printf("\n");
		for (int i=0; i<(sizeof(AT_GraphPerf)/sizeof(unsigned int)); i++) {
			printf("\t\t\t Layer %d: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", i, AT_GraphPerf[i], AT_GraphOperInfosNames[i], ((float) AT_GraphOperInfosNames[i])/ AT_GraphPerf[i]);
			TotalCycles += AT_GraphPerf[i]; TotalOper += AT_GraphOperInfosNames[i];
		}
		printf("\n");
		printf("\t\t\t %s: Cycles: %10d, Operations: %10d, Operations/Cycle: %f\n", "Total", TotalCycles, TotalOper, ((float) TotalOper)/ TotalCycles);
		printf("\n");
	}
#endif
	
#ifndef __EMUL__
	pmsis_exit(0);
#endif
	printf("Ended\n");
	return 0;
}

#ifndef __EMUL__
int main(void)
{
    printf("\n\n\t *** Visualwakewords for vehicle ***\n\n");
    return pmsis_kickoff((void *) body);
}
#endif
