/*
 * Copyright (C) 2017 GreenWaves Technologies
 * All rights reserved.
 *
 * This software may be modified and distributed under the terms
 * of the BSD license.  See the LICENSE file for details.
 *
 */

#include <stdio.h>

#include "mbv1_grayscale.h"
#include "mbv1_grayscaleKernels.h"
#include "ImgIO.h"

#include <string.h>
#include <dirent.h>

#define __XSTR(__s) __STR(__s)
#define __STR(__s) #__s


#ifdef PERF
#undef PERF
#endif

#define AT_IN_SIZE (AT_IN_WIDTH*AT_IN_HEIGHT*AT_IN_COLORS)

#ifndef STACK_SIZE
#define STACK_SIZE     2048 
#endif

AT_HYPERFLASH_FS_EXT_ADDR_TYPE __PREFIX(_L3_Flash);


// Softmax always outputs Q15 short int even from 8 bit input
L2_MEM short int *ResOut;
L2_MEM unsigned char imgin_unsigned[AT_IN_SIZE];
L2_MEM signed char *imgin_signed = imgin_unsigned;

static int RunNetwork()
{
  __PREFIX(CNN)(imgin_signed, ResOut);
  //Checki Results
  /*if (ResOut[1] > ResOut[0]) {
    printf("person seen (%d, %d)\n", ResOut[0], ResOut[1]);
  } else {
    printf("no person seen (%d, %d)\n", ResOut[0], ResOut[1]);
  }
  printf("\n");*/
  return (ResOut[1] > ResOut[0]);
}

#define MAXCHAR 1000
#define NUM_CLASS 2
//#define DATASET_DIM 10
int main(int argc, char *argv[]) 
{
  if (argc < 2) {
    printf("Usage: %s [image_file]\n", argv[0]);
    exit(1);
  }

/* TO READ LABELS FROM FILE */
/*FILE *fp;
  char str[MAXCHAR];
  char* filename = "./label.txt";
  int init_size;

  fp = fopen(filename, "r");
  if (fp == NULL){
      printf("Could not open file %s",filename);
  }
  while (fgets(str, MAXCHAR, fp) != NULL){  
      init_size = strlen(str);
      printf("%s ---- %d\n", str, init_size);
  }
  fclose(fp);
*/ 
  struct dirent *dp;
  DIR *dfd;

  char *dir = argv[1];

  printf("Entering main controller\n");

  printf("Constructor\n");

  // IMPORTANT - MUST BE CALLED AFTER THE CLUSTER IS SWITCHED ON!!!!
  if (__PREFIX(CNN_Construct)())
  {
    printf("Graph constructor exited with an error\n");
    return 1;
  }
  char filename_qfd[100];
  char new_name_qfd[100];
  int label;
  int i=0;
  int predicted=0;
  int false_positive=0;
  int num_neg = 0;
  int num_pos = 0;
  int false_negative=0;
  int result;


/*--------------------iterate over files in dir------------------------*/
  if ((dfd = opendir(dir)) == NULL)
  {
    fprintf(stderr, "Can't open %s\n", dir);
    return 0;
  }
  while ((dp = readdir(dfd)) != NULL){
    struct stat stbuf ;
    sprintf( filename_qfd , "%s/%s" ,dir,dp->d_name);
    if( stat(filename_qfd,&stbuf ) == -1 ){
     printf("Unable to stat file: %s\n",filename_qfd) ;
     continue ;
    }

    if ( ( stbuf.st_mode & S_IFMT ) == S_IFDIR ){
     continue;
     // Skip directories
    } else {
      label = dp->d_name[strlen(dp->d_name)-5] - '0'; 
      //Reading Image from Bridge
      if (ReadImageFromFile(filename_qfd, AT_IN_WIDTH, AT_IN_HEIGHT, 1, imgin_unsigned, AT_IN_SIZE*sizeof(unsigned char), 0, 0)) {
        printf("Failed to load image %s\n", filename_qfd);
        return 1;
      }
      /*--------------------convert to signed in [-128:127]----------------*/
      for(int i=0; i<AT_IN_SIZE; i++){
        imgin_signed[i] = (signed char) ( ((int) (imgin_unsigned[i])) - 128);
      }
      #ifdef PRINT_IMAGE
        for (int i=0; i<AT_IN_SIZE; i++) {
            printf("%03d, ", __PREFIX(_L2_Memory[i]));
          }
          printf("\n");
      #endif
      /*----------------------Allocate output-----------------------------*/
      ResOut = (short int *) AT_L2_ALLOC(0, NUM_CLASS*sizeof(short int));
      if (ResOut==0) {
        printf("Failed to allocate Memory for Result (%ld bytes)\n", NUM_CLASS*sizeof(short int));
        return 1;
      }

      /*------------------Execute the function "RunNetwork"--------------*/
      result = RunNetwork(NULL);
      if (result==label){
        predicted++;
      } else {
        if (result){
          false_positive++;
        } else {
          false_negative++;
        }
      }
    }
    num_pos += label;
    num_neg += !label;
    i++;
    if (!(i%50)){
      printf("%d/40504 \n %.2f%% \n false_negative: %d/%d \n false_positive: %d/%d \n", \
        i, 100*(float)predicted/(float)i, false_negative, num_pos, false_positive, num_neg);
    }
  }
  printf("accuracy rate = %.2f%%\n", 100*(float)predicted/(float)i);
  
  __PREFIX(CNN_Destruct)();

  printf("Ended\n");
  return 0;
}

