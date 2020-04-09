#include <stdint.h>
#include <stdio.h>
#include "AutoTilerLib.h"
#include "CNN_Generators.h"


void mbv1_grayscaleModel(unsigned int L1Memory, unsigned int L2Memory, unsigned int L3Memory, unsigned int L3Flash)
{
    KernelOper_T Cop = KOP_CONV;

    // SetKernelOpts(KER_OPT_NONE, KER_OPT_BUFFER_PROMOTE);
    SetSymbolDynamics();

    SetUsedFilesNames(0, 2, "CNN_BasicKernels.h", "mbv1_grayscale.h");
    SetGeneratedFilesNames("mbv1_grayscaleKernels.c", "mbv1_grayscaleKernels.h");
    AT_SetGraphCtrl(AT_GRAPH_MONITOR_CYCLES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_NODE_NAMES, AT_OPT_ON);
    AT_SetGraphCtrl(AT_GRAPH_PRODUCE_OPERINFOS, AT_OPT_ON);

    SetMemoryDeviceInfos(4,
        AT_MEM_L1, L1Memory, "mbv1_grayscale_L1_Memory", 0, 0,
        AT_MEM_L2, L2Memory, "mbv1_grayscale_L2_Memory", 0, 1,
        AT_MEM_L3_HRAM, L3Memory, "mbv1_grayscale_L3_Memory", 0, 0,
        AT_MEM_L3_HFLASH, L3Flash, "mbv1_grayscale_L3_Flash", "mbv1_grayscale_L3_Flash_Const.dat", 0
    );

    LoadCNNLibrary();

    // generator for DEPTHWISE_CONV_2D_0_0_fusion
    CNN_GenControl_T gen_ctrl_S1_Conv2d_32x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S1_Conv2d_32x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S1_Conv2d_32x1x3x3_Relu6, "PADTYPE", (void *)1);
    CNN_ConvolutionPoolReLU("S1_Conv2d_32x1x3x3_Relu6", &gen_ctrl_S1_Conv2d_32x1x3x3_Relu6, 1, 1, 1, 1, 7, 4, 7, 4, 1, 1, 1, 1, 1, 32, 224, 224,
        KOP_CONV_DP, 3, 3, 1, 1, 2, 2, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_1_fusion
    CNN_ConvolutionPoolReLU("S2_Conv2d_32x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 5, 6, 4, 1, 1, 1, 1, 32, 32, 112, 112,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_2_fusion
    CNN_ConvolutionPoolReLU("S3_Conv2d_64x32x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 32, 64, 112, 112,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_3_fusion
    CNN_GenControl_T gen_ctrl_S4_Conv2d_64x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S4_Conv2d_64x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S4_Conv2d_64x1x3x3_Relu6, "PADTYPE", (void *)1);
    CNN_ConvolutionPoolReLU("S4_Conv2d_64x1x3x3_Relu6", &gen_ctrl_S4_Conv2d_64x1x3x3_Relu6, 1, 1, 1, 1, 4, 6, 5, 4, 1, 1, 1, 1, 64, 64, 112, 112,
        KOP_CONV_DWDP, 3, 3, 1, 1, 2, 2, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_4_fusion
    CNN_ConvolutionPoolReLU("S5_Conv2d_128x64x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 64, 128, 56, 56,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_5_fusion
    CNN_ConvolutionPoolReLU("S6_Conv2d_128x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 5, 6, 4, 1, 1, 1, 1, 128, 128, 56, 56,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_6_fusion
    CNN_ConvolutionPoolReLU("S7_Conv2d_128x128x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 128, 128, 56, 56,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_7_fusion
    CNN_GenControl_T gen_ctrl_S8_Conv2d_128x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S8_Conv2d_128x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S8_Conv2d_128x1x3x3_Relu6, "PADTYPE", (void *)1);
    CNN_ConvolutionPoolReLU("S8_Conv2d_128x1x3x3_Relu6", &gen_ctrl_S8_Conv2d_128x1x3x3_Relu6, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 128, 128, 56, 56,
        KOP_CONV_DWDP, 3, 3, 1, 1, 2, 2, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_8_fusion
    CNN_ConvolutionPoolReLU("S9_Conv2d_256x128x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 128, 256, 28, 28,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_9_fusion
    CNN_ConvolutionPoolReLU("S10_Conv2d_256x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 256, 256, 28, 28,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_10_fusion
    CNN_ConvolutionPoolReLU("S11_Conv2d_256x256x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 256, 256, 28, 28,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_11_fusion
    CNN_GenControl_T gen_ctrl_S12_Conv2d_256x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S12_Conv2d_256x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S12_Conv2d_256x1x3x3_Relu6, "PADTYPE", (void *)1);
    CNN_ConvolutionPoolReLU("S12_Conv2d_256x1x3x3_Relu6", &gen_ctrl_S12_Conv2d_256x1x3x3_Relu6, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 256, 256, 28, 28,
        KOP_CONV_DWDP, 3, 3, 1, 1, 2, 2, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_12_fusion
    CNN_ConvolutionPoolReLU("S13_Conv2d_512x256x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 256, 512, 14, 14,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_13_fusion
    CNN_ConvolutionPoolReLU("S14_Conv2d_512x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_14_fusion
    CNN_ConvolutionPoolReLU("S15_Conv2d_512x512x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_15_fusion
    CNN_ConvolutionPoolReLU("S16_Conv2d_512x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_16_fusion
    CNN_ConvolutionPoolReLU("S17_Conv2d_512x512x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_17_fusion
    CNN_ConvolutionPoolReLU("S18_Conv2d_512x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_18_fusion
    CNN_ConvolutionPoolReLU("S19_Conv2d_512x512x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_19_fusion
    CNN_ConvolutionPoolReLU("S20_Conv2d_512x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_20_fusion
    CNN_ConvolutionPoolReLU("S21_Conv2d_512x512x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_21_fusion
    CNN_ConvolutionPoolReLU("S22_Conv2d_512x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 7, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_22_fusion
    CNN_ConvolutionPoolReLU("S23_Conv2d_512x512x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_23_fusion
    CNN_GenControl_T gen_ctrl_S24_Conv2d_512x1x3x3_Relu6;
    CNN_InitGenCtrl(&gen_ctrl_S24_Conv2d_512x1x3x3_Relu6);
    CNN_SetGenCtrl(&gen_ctrl_S24_Conv2d_512x1x3x3_Relu6, "PADTYPE", (void *)1);
    CNN_ConvolutionPoolReLU("S24_Conv2d_512x1x3x3_Relu6", &gen_ctrl_S24_Conv2d_512x1x3x3_Relu6, 1, 1, 1, 1, 4, 6, 7, 4, 1, 1, 1, 1, 512, 512, 14, 14,
        KOP_CONV_DWDP, 3, 3, 1, 1, 2, 2, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_24_fusion
    CNN_ConvolutionPoolReLU("S25_Conv2d_1024x512x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 512, 1024, 7, 7,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for DEPTHWISE_CONV_2D_0_25_fusion
    CNN_ConvolutionPoolReLU("S26_Conv2d_1024x1x3x3_Relu6", 0, 1, 1, 1, 1, 4, 6, 6, 4, 1, 1, 1, 1, 1024, 1024, 7, 7,
        KOP_CONV_DWDP, 3, 3, 1, 1, 1, 1, 1,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for CONV_2D_0_26_fusion
    CNN_ConvolutionPoolReLU("S27_Conv2d_1024x1024x1x1_Relu6", 0, 1, 1, 1, 1, 4, 7, 6, 4, 1, 1, 1, 1, 1024, 1024, 7, 7,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_RELUN);
    // generator for AVERAGE_POOL_2D_0_27
    CNN_PoolReLU("S28_AveragePool_7x7", 0, 1, 1, 4, 4, 1, 1, 1024, 1024, 7, 7,
        KOP_AVGPOOL, 7, 7, 1, 1, 2, 2, 0, KOP_NONE);
    // generator for CONV_2D_0_28
    CNN_ConvolutionPoolReLU("S29_Conv2d_2x1024x1x1", 0, 1, 1, 1, 1, 4, 7, 7, 4, 1, 1, 1, 1, 1024, 2, 1, 1,
        KOP_CONV_DP, 1, 1, 1, 1, 1, 1, 0,
        KOP_NONE, 0, 0, 0, 0, 0, 0, 0, KOP_NONE);
    // generator for SOFTMAX_0_30
    CNN_SoftMax("S31_SoftMax", 0, 1, 2, 4, 15, 1, 1, 2, KOP_SOFTMAX);

#define GRAPH
#ifdef GRAPH
    CreateGraph("mbv1_grayscaleCNN",
        /* Arguments either passed or globals */
        CArgs(58,
            TCArgInfo("signed char *__restrict__", "Input_1", ARG_SCOPE_ARG, ARG_DIR_IN, AT_MEM_L2, AT_MEM_L2, 0),
            TCArgInfo("signed char *__restrict__", "Step1Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step1Weights.tensor", 1, 1, 8, 4)),
            TCArgInfo("signed char *__restrict__", "Step1Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step1Biases.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step2Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step2Weights.tensor", 1, 1, 8, 5)),
            TCArgInfo("signed char *__restrict__", "Step2Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step2Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step3Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step3Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step3Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step3Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step4Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step4Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step4Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step4Biases.tensor", 1, 1, 8, 5)),
            TCArgInfo("signed char *__restrict__", "Step5Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step5Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step5Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step5Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step6Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step6Weights.tensor", 1, 1, 8, 5)),
            TCArgInfo("signed char *__restrict__", "Step6Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step6Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step7Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step7Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step7Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step7Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step8Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step8Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step8Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step8Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step9Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step9Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step9Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step9Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step10Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step10Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step10Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step10Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step11Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step11Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step11Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step11Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step12Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step12Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step12Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step12Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step13Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step13Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step13Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step13Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step14Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step14Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step14Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step14Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step15Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step15Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step15Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step15Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step16Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step16Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step16Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step16Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step17Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step17Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step17Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step17Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step18Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step18Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step18Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step18Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step19Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step19Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step19Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step19Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step20Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step20Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step20Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step20Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step21Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step21Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step21Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step21Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step22Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step22Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step22Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step22Biases.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step23Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step23Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step23Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step23Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step24Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step24Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step24Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step24Biases.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step25Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step25Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step25Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step25Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step26Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step26Weights.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step26Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step26Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step27Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step27Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step27Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step27Biases.tensor", 1, 1, 8, 6)),
            TCArgInfo("signed char *__restrict__", "Step29Weights", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step29Weights.tensor", 1, 1, 8, 7)),
            TCArgInfo("signed char *__restrict__", "Step29Biases", ARG_SCOPE_GLOBAL, ARG_DIR_CONSTIN, AT_MEM_L3_HFLASH, AT_MEM_UNDEF, ConstInfo("GAP_PORTING_8BIT/tensors/Step29Biases.tensor", 1, 1, 8, 7)),
            TCArgInfo("short int *__restrict__", "Output_1", ARG_SCOPE_ARG, ARG_DIR_OUT, AT_MEM_L2, AT_MEM_L2, 0)
        ),
        /* Locals, allocated dynamically */
        CArgs(29,
            TCArgInfo("signed char *__restrict__", "OutputStep1", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep2", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep3", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep4", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep5", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep6", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep7", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep8", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep9", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep10", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep11", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep12", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep13", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep14", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep15", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep16", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep17", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep18", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep19", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep20", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep21", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep22", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep23", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep24", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep25", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep26", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep27", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep28", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0),
            TCArgInfo("signed char *__restrict__", "OutputStep29", ARG_SCOPE_LOCAL, ARG_DIR_INOUT, AT_MEM_UNDEF, AT_MEM_UNDEF, 0)
        )
    );

    /* Stacked tensors - Concats */
    // no concats in graph so not stacked tensors created

    // Node S1_Conv2d_32x1x3x3_Relu6 inq 7 weightsq 4 outq 4 biasesq 7
    AddNode("S1_Conv2d_32x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "Input_1", 0), GNodeArg(GNA_IN, "Step1Weights", 0), GNodeArg(GNA_IN, "Step1Biases", 0), GNodeArg(GNA_OUT, "OutputStep1", 0)));
    // Node S2_Conv2d_32x1x3x3_Relu6 inq 4 weightsq 5 outq 4 biasesq 6
    AddNode("S2_Conv2d_32x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep1", 0), GNodeArg(GNA_IN, "Step2Weights", 0), GNodeArg(GNA_IN, "Step2Biases", 0), GNodeArg(GNA_OUT, "OutputStep2", 0)));
    // Node S3_Conv2d_64x32x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S3_Conv2d_64x32x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep2", 0), GNodeArg(GNA_IN, "Step3Weights", 0), GNodeArg(GNA_IN, "Step3Biases", 0), GNodeArg(GNA_OUT, "OutputStep3", 0)));
    // Node S4_Conv2d_64x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 5
    AddNode("S4_Conv2d_64x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep3", 0), GNodeArg(GNA_IN, "Step4Weights", 0), GNodeArg(GNA_IN, "Step4Biases", 0), GNodeArg(GNA_OUT, "OutputStep4", 0)));
    // Node S5_Conv2d_128x64x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S5_Conv2d_128x64x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep4", 0), GNodeArg(GNA_IN, "Step5Weights", 0), GNodeArg(GNA_IN, "Step5Biases", 0), GNodeArg(GNA_OUT, "OutputStep5", 0)));
    // Node S6_Conv2d_128x1x3x3_Relu6 inq 4 weightsq 5 outq 4 biasesq 6
    AddNode("S6_Conv2d_128x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep5", 0), GNodeArg(GNA_IN, "Step6Weights", 0), GNodeArg(GNA_IN, "Step6Biases", 0), GNodeArg(GNA_OUT, "OutputStep6", 0)));
    // Node S7_Conv2d_128x128x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S7_Conv2d_128x128x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep6", 0), GNodeArg(GNA_IN, "Step7Weights", 0), GNodeArg(GNA_IN, "Step7Biases", 0), GNodeArg(GNA_OUT, "OutputStep7", 0)));
    // Node S8_Conv2d_128x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S8_Conv2d_128x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep7", 0), GNodeArg(GNA_IN, "Step8Weights", 0), GNodeArg(GNA_IN, "Step8Biases", 0), GNodeArg(GNA_OUT, "OutputStep8", 0)));
    // Node S9_Conv2d_256x128x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S9_Conv2d_256x128x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep8", 0), GNodeArg(GNA_IN, "Step9Weights", 0), GNodeArg(GNA_IN, "Step9Biases", 0), GNodeArg(GNA_OUT, "OutputStep9", 0)));
    // Node S10_Conv2d_256x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S10_Conv2d_256x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep9", 0), GNodeArg(GNA_IN, "Step10Weights", 0), GNodeArg(GNA_IN, "Step10Biases", 0), GNodeArg(GNA_OUT, "OutputStep10", 0)));
    // Node S11_Conv2d_256x256x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S11_Conv2d_256x256x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep10", 0), GNodeArg(GNA_IN, "Step11Weights", 0), GNodeArg(GNA_IN, "Step11Biases", 0), GNodeArg(GNA_OUT, "OutputStep11", 0)));
    // Node S12_Conv2d_256x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S12_Conv2d_256x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep11", 0), GNodeArg(GNA_IN, "Step12Weights", 0), GNodeArg(GNA_IN, "Step12Biases", 0), GNodeArg(GNA_OUT, "OutputStep12", 0)));
    // Node S13_Conv2d_512x256x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S13_Conv2d_512x256x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep12", 0), GNodeArg(GNA_IN, "Step13Weights", 0), GNodeArg(GNA_IN, "Step13Biases", 0), GNodeArg(GNA_OUT, "OutputStep13", 0)));
    // Node S14_Conv2d_512x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S14_Conv2d_512x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep13", 0), GNodeArg(GNA_IN, "Step14Weights", 0), GNodeArg(GNA_IN, "Step14Biases", 0), GNodeArg(GNA_OUT, "OutputStep14", 0)));
    // Node S15_Conv2d_512x512x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S15_Conv2d_512x512x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep14", 0), GNodeArg(GNA_IN, "Step15Weights", 0), GNodeArg(GNA_IN, "Step15Biases", 0), GNodeArg(GNA_OUT, "OutputStep15", 0)));
    // Node S16_Conv2d_512x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S16_Conv2d_512x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep15", 0), GNodeArg(GNA_IN, "Step16Weights", 0), GNodeArg(GNA_IN, "Step16Biases", 0), GNodeArg(GNA_OUT, "OutputStep16", 0)));
    // Node S17_Conv2d_512x512x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S17_Conv2d_512x512x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep16", 0), GNodeArg(GNA_IN, "Step17Weights", 0), GNodeArg(GNA_IN, "Step17Biases", 0), GNodeArg(GNA_OUT, "OutputStep17", 0)));
    // Node S18_Conv2d_512x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S18_Conv2d_512x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep17", 0), GNodeArg(GNA_IN, "Step18Weights", 0), GNodeArg(GNA_IN, "Step18Biases", 0), GNodeArg(GNA_OUT, "OutputStep18", 0)));
    // Node S19_Conv2d_512x512x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S19_Conv2d_512x512x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep18", 0), GNodeArg(GNA_IN, "Step19Weights", 0), GNodeArg(GNA_IN, "Step19Biases", 0), GNodeArg(GNA_OUT, "OutputStep19", 0)));
    // Node S20_Conv2d_512x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S20_Conv2d_512x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep19", 0), GNodeArg(GNA_IN, "Step20Weights", 0), GNodeArg(GNA_IN, "Step20Biases", 0), GNodeArg(GNA_OUT, "OutputStep20", 0)));
    // Node S21_Conv2d_512x512x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S21_Conv2d_512x512x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep20", 0), GNodeArg(GNA_IN, "Step21Weights", 0), GNodeArg(GNA_IN, "Step21Biases", 0), GNodeArg(GNA_OUT, "OutputStep21", 0)));
    // Node S22_Conv2d_512x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 7
    AddNode("S22_Conv2d_512x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep21", 0), GNodeArg(GNA_IN, "Step22Weights", 0), GNodeArg(GNA_IN, "Step22Biases", 0), GNodeArg(GNA_OUT, "OutputStep22", 0)));
    // Node S23_Conv2d_512x512x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S23_Conv2d_512x512x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep22", 0), GNodeArg(GNA_IN, "Step23Weights", 0), GNodeArg(GNA_IN, "Step23Biases", 0), GNodeArg(GNA_OUT, "OutputStep23", 0)));
    // Node S24_Conv2d_512x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 7
    AddNode("S24_Conv2d_512x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep23", 0), GNodeArg(GNA_IN, "Step24Weights", 0), GNodeArg(GNA_IN, "Step24Biases", 0), GNodeArg(GNA_OUT, "OutputStep24", 0)));
    // Node S25_Conv2d_1024x512x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S25_Conv2d_1024x512x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep24", 0), GNodeArg(GNA_IN, "Step25Weights", 0), GNodeArg(GNA_IN, "Step25Biases", 0), GNodeArg(GNA_OUT, "OutputStep25", 0)));
    // Node S26_Conv2d_1024x1x3x3_Relu6 inq 4 weightsq 6 outq 4 biasesq 6
    AddNode("S26_Conv2d_1024x1x3x3_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep25", 0), GNodeArg(GNA_IN, "Step26Weights", 0), GNodeArg(GNA_IN, "Step26Biases", 0), GNodeArg(GNA_OUT, "OutputStep26", 0)));
    // Node S27_Conv2d_1024x1024x1x1_Relu6 inq 4 weightsq 7 outq 4 biasesq 6
    AddNode("S27_Conv2d_1024x1024x1x1_Relu6", Bindings(4, GNodeArg(GNA_IN, "OutputStep26", 0), GNodeArg(GNA_IN, "Step27Weights", 0), GNodeArg(GNA_IN, "Step27Biases", 0), GNodeArg(GNA_OUT, "OutputStep27", 0)));
    // Node AVERAGE_POOL_2D_0_27 inq 4 outq 4
    AddNode("S28_AveragePool_7x7", Bindings(2, GNodeArg(GNA_IN, "OutputStep27", 0), GNodeArg(GNA_OUT, "OutputStep28", 0)));
    // Node S29_Conv2d_2x1024x1x1 inq 4 weightsq 7 outq 4 biasesq 7
    AddNode("S29_Conv2d_2x1024x1x1", Bindings(4, GNodeArg(GNA_IN, "OutputStep28", 0), GNodeArg(GNA_IN, "Step29Weights", 0), GNodeArg(GNA_IN, "Step29Biases", 0), GNodeArg(GNA_OUT, "OutputStep29", 0)));
    // Node SOFTMAX_0_30 inq 4 outq 15
    AddNode("S31_SoftMax", Bindings(2, GNodeArg(GNA_IN, "OutputStep29", 0), GNodeArg(GNA_OUT, "Output_1", 0)));
    CloseGraph();
#endif
}

int main(int argc, char **argv)

{
    if (TilerParseOptions(argc, argv)) {
            printf("Failed to initialize or incorrect output arguments directory.\n"); return 1;
    }
    mbv1_grayscaleModel(52000, 300*1024, 8*1024*1024, 20*1024*1024);
    GenerateTilingCode();
    return 0;
}
